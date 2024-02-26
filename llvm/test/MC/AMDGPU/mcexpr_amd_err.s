// RUN: not llvm-mc -triple amdgcn-amd-amdhsa %s 2>&1 | FileCheck --check-prefix=ASM %s

// ASM: 20:22: error: empty max expression
// ASM: 20:22: error: missing expression
// ASM: 21:20: error: empty or expression
// ASM: 21:20: error: missing expression
// ASM: 23:29: error: unknown token in expression
// ASM: 23:29: error: missing expression
// ASM: 24:32: error: unexpected token in max expression
// ASM: 24:32: error: missing expression
// ASM: 25:42: error: unknown token in expression
// ASM: 25:42: error: missing expression
// ASM: 26:38: error: unexpected token in or expression
// ASM: 26:38: error: missing expression

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
