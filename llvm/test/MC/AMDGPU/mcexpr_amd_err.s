// RUN: not llvm-mc -triple amdgcn-amd-amdhsa %s 2>&1 | FileCheck --check-prefix=ASM %s

.set one, 1
.set two, 2
.set three, 3

.set max_empty, max()
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: empty max expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set or_empty, or()
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: empty or expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set max_post_aux_comma, max(one,)
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: mismatch of commas in max expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set max_pre_aux_comma, max(,one)
// asm: :[[@line-1]]:{{[0-9]+}}: error: unknown token in expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set max_double_comma, max(one,, two)
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: unknown token in expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set max_no_comma, max(one two)
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: unexpected token in max expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set max_missing_paren, max(two
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: unexpected token in max expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set max_expression_one, max(three, four,
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: unknown token in expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set or_expression_one, or(four, five
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: unexpected token in or expression
// ASM: :[[@LINE-2]]:{{[0-9]+}}: error: missing expression

.set max_no_lparen, max four, five)
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: expected newline

.set max_no_paren, max one, two, three
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: expected newline

.set max_rparen_only, max)
// ASM: :[[@LINE-1]]:{{[0-9]+}}: error: expected newline

.set four, 4
.set five, 5
