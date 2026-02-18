# Check that the "6 - 3b <= a" constraint is simplified to "b >= 2"
# in the generated AST.
domain: "[a,b] -> { S[] : 0 <= a <= 2 and 6 - 3b <= a }"
