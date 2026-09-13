from lit.BooleanExpression import BooleanExpression


class _Parser(BooleanExpression):
    """Reuse lit's tokens and grammar, but retain the expression's structure."""

    def parseMATCH(self):
        if "{{" in self.token:
            raise ValueError("--filter-requires does not support '{{regex}}' patterns")
        self.value = ("literal", self.token)
        self.token = next(self.tokens)

    def parseNOT(self):
        if self.accept("!"):
            self.parseNOT()
            self.value = ("not", self.value)
        else:
            super().parseNOT()

    def parseAND(self):
        self.parseNOT()
        children = [self.value]
        while self.accept("&&"):
            self.parseNOT()
            children.append(self.value)
        self.value = ("and", children)

    def parseOR(self):
        self.parseAND()
        children = [self.value]
        while self.accept("||"):
            self.parseAND()
            children.append(self.value)
        self.value = ("or", children)


class FilterRequires:
    """Select tests by their required features, not available features.

    Expressions are expanded into combinations of required and excluded
    features. A test combination matches a filter combination when their
    required features are identical and the test adds no extra excluded
    features. Contradictory combinations are discarded.
    """

    MAX_COMBINATIONS = 1024
    MAX_STEPS = 100000

    def __init__(self, expression):
        self.expression = expression
        self.is_base = expression.strip() == "Base"
        self.combinations = None if self.is_base else self._normalize([expression])

    def __str__(self):
        return self.expression

    @classmethod
    def _normalize(cls, expressions):
        steps = 0

        def step():
            nonlocal steps
            steps += 1
            if steps > cls.MAX_STEPS:
                raise ValueError(
                    "--filter-requires expression exceeds the normalization "
                    "limit (%d steps)" % cls.MAX_STEPS
                )

        def add(combinations, combination):
            combinations.add(combination)
            if len(combinations) > cls.MAX_COMBINATIONS:
                raise ValueError(
                    "--filter-requires expression exceeds the combination "
                    "limit (%d combinations)" % cls.MAX_COMBINATIONS
                )

        def combine(left, right):
            result = set()
            for lp, ln in left:
                for rp, rn in right:
                    step()
                    if not (lp & rn or rp & ln):
                        add(result, (lp | rp, ln | rn))
            return result

        def expand(node, negate=False):
            step()
            kind, value = node
            if kind == "literal":
                if value == "true":
                    return set() if negate else {empty}
                literal = frozenset([value])
                return {(empty[0], literal) if negate else (literal, empty[1])}
            if kind == "not":
                return expand(value, not negate)
            # Push NOT through AND/OR using De Morgan's laws.
            conjunction = (kind == "and") != negate
            combinations = {empty} if conjunction else set()
            for child in value:
                child_combinations = expand(child, negate)
                if conjunction:
                    combinations = combine(combinations, child_combinations)
                else:
                    for combination in child_combinations:
                        step()
                        add(combinations, combination)
            return combinations

        empty = (frozenset(), frozenset())
        result = {empty}
        try:
            for expression in expressions:
                # Commas delimit complete expressions, as in REQUIRES lines.
                for part in expression.split(","):
                    tree = _Parser(part.strip(), set()).parseAll()
                    result = combine(result, expand(tree))
        except RecursionError:
            raise ValueError(
                "--filter-requires expression is nested too deeply"
            ) from None
        except ValueError as error:
            raise ValueError("%s\nin REQUIRES selection: %r" % (error, expression))
        return result

    def matches(self, requirements):
        if self.is_base:
            return not requirements
        if not requirements:
            return False
        test_combinations = self._normalize(requirements)
        return any(
            test_positive == caller_positive and test_negative <= caller_negative
            for test_positive, test_negative in test_combinations
            for caller_positive, caller_negative in self.combinations
        )
