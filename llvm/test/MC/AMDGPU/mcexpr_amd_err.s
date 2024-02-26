// RUN: not llvm-mc -triple amdgcn-amd-amdhsa %s 2>&1 | FileCheck --check-prefix=ASM %s

// ASM: error: empty max expression
// ASM: error: missing expression
// ASM: error: empty or expression
// ASM: error: missing expression
// ASM: error: unknown token in expression
// ASM: error: missing expression
// ASM: error: unexpected token in max expression
// ASM: error: missing expression
// ASM: error: unknown token in expression
// ASM: error: missing expression
// ASM: error: unexpected token in or expression
// ASM: error: missing expression

.set one, 1
.set two, 2
.set three, 3

.set max_empty, max()
.set or_empty, or()
.set max_post_aux_comma, max(one,)
.set max_pre_aux_comma, max(,one)
.set max_missing_paren, max(two
.set max_expression_one, max(three, four,
.set or_expression_one, or(four, five

.set four, 4
.set five, 5
