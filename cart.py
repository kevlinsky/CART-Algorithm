header = ['SeriousDlqin2yrs','RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']


def unique_values(rows, column):
    return set([row[column] for row in rows])


def classes_count(rows):
    classes = {}
    for row in rows:
        label = row[-1]
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
    return classes


def is_number(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, row):
        value = row[self.column]
        if is_number(value):
            return value >= self.value
        else:
            return value == self.value

    def __repr__(self):
        condition = "=="
        if is_number(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


def partition(rows, question):
    true_rows = []
    false_rows = []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    classes = classes_count(rows)
    impurity = 1
    for label in classes:
        probability_of_label = classes[label] / float(len(rows))
        impurity -= probability_of_label ** 2
    return impurity


def information_gain(left, right, current_uncertainty):
    probability = float(len(left)) / float(len(right) + len(right))
    return current_uncertainty - probability * gini(left) - (1 - probability) * gini(right)


def find_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainly = gini(rows)
    number_of_features = len(rows[0]) - 1
    for column in range(number_of_features):
        u_values = set([row[column] for row in rows])
        for value in u_values:
            question = Question(column, value)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = information_gain(true_rows, false_rows, current_uncertainly)
            if gain >= best_gain:
                best_gain = gain
                best_question = question
    return best_gain, best_question


class Leaf:
    def __init__(self, rows):
        self.predictions = classes_count(rows)


class Decision:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(classes):
    total = sum(classes.values()) * 1.0
    probabilities = {}
    for label in classes.keys():
        probabilities[label] = str(int(classes[label] / total * 100)) + "%"
    return probabilities


def read_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        rows = f.readlines()
        rows = list(map(lambda x: x.strip(), rows))
        for row in rows:
            row = row.split(',')
            data.append(row)
    return data


if __name__ == "__main__":
    tree = build_tree(read_data('training_data.csv'))
    print_tree(tree)
    testing_data = read_data('test_data.csv')
    for row in testing_data:
        print("Actual: %s. Predicted: %s" % (row[-1], print_leaf(classify(row, tree))))