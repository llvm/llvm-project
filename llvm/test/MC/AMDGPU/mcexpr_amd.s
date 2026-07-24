// RUN: llvm-mc -triple=amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -filetype=obj < %s > %t
// RUN: llvm-objdump --syms %t | FileCheck --check-prefix=OBJDUMP %s

// OBJDUMP: SYMBOL TABLE:
// OBJDUMP-NEXT: 0000000000000000 l       *ABS*  0000000000000000 zero
// OBJDUMP-NEXT: 0000000000000001 l       *ABS*  0000000000000000 one
// OBJDUMP-NEXT: 0000000000000002 l       *ABS*  0000000000000000 two
// OBJDUMP-NEXT: 0000000000000003 l       *ABS*  0000000000000000 three
// OBJDUMP-NEXT: 7fffffffffffffff l       *ABS*  0000000000000000 i64_max
// OBJDUMP-NEXT: 8000000000000000 l       *ABS*  0000000000000000 i64_min
// OBJDUMP-NEXT: 0000000000000005 l       *ABS*  0000000000000000 max_expression_all
// OBJDUMP-NEXT: 0000000000000005 l       *ABS*  0000000000000000 five
// OBJDUMP-NEXT: 0000000000000004 l       *ABS*  0000000000000000 four
// OBJDUMP-NEXT: 0000000000000002 l       *ABS*  0000000000000000 max_expression_two
// OBJDUMP-NEXT: 0000000000000001 l       *ABS*  0000000000000000 max_expression_one
// OBJDUMP-NEXT: 000000000000000a l       *ABS*  0000000000000000 max_literals
// OBJDUMP-NEXT: 000000000000000f l       *ABS*  0000000000000000 max_with_max_sym
// OBJDUMP-NEXT: 000000000000000f l       *ABS*  0000000000000000 max
// OBJDUMP-NEXT: ffffffffffffffff l       *ABS*  0000000000000000 max_neg_numbers
// OBJDUMP-NEXT: ffffffffffffffff l       *ABS*  0000000000000000 max_neg_number
// OBJDUMP-NEXT: 0000000000000003 l       *ABS*  0000000000000000 max_with_subexpr
// OBJDUMP-NEXT: 0000000000000006 l       *ABS*  0000000000000000 max_as_subexpr
// OBJDUMP-NEXT: 0000000000000005 l       *ABS*  0000000000000000 max_recursive_subexpr
// OBJDUMP-NEXT: 7fffffffffffffff l       *ABS*  0000000000000000 max_expr_one_max
// OBJDUMP-NEXT: 7fffffffffffffff l       *ABS*  0000000000000000 max_expr_two_max
// OBJDUMP-NEXT: 7fffffffffffffff l       *ABS*  0000000000000000 max_expr_three_max
// OBJDUMP-NEXT: 8000000000000000 l       *ABS*  0000000000000000 max_expr_one_min
// OBJDUMP-NEXT: 0000000000000003 l       *ABS*  0000000000000000 max_expr_two_min
// OBJDUMP-NEXT: 0000000000989680 l       *ABS*  0000000000000000 max_expr_three_min
// OBJDUMP-NEXT: 0000000000000001 l       *ABS*  0000000000000000 min_expression_all
// OBJDUMP-NEXT: 0000000000000001 l       *ABS*  0000000000000000 min_expression_two
// OBJDUMP-NEXT: 0000000000000003 l       *ABS*  0000000000000000 min_expression_one
// OBJDUMP-NEXT: 0000000000000001 l       *ABS*  0000000000000000 min_literals
// OBJDUMP-NEXT: 0000000000000000 l       *ABS*  0000000000000000 min_with_min_sym
// OBJDUMP-NEXT: 0000000000000000 l       *ABS*  0000000000000000 min
// OBJDUMP-NEXT: ffffffffffffffff l       *ABS*  0000000000000000 neg_one
// OBJDUMP-NEXT: fffffffffffffffb l       *ABS*  0000000000000000 min_neg_numbers
// OBJDUMP-NEXT: ffffffffffffffff l       *ABS*  0000000000000000 min_neg_number
// OBJDUMP-NEXT: 0000000000000003 l       *ABS*  0000000000000000 min_with_subexpr
// OBJDUMP-NEXT: 0000000000000004 l       *ABS*  0000000000000000 min_as_subexpr
// OBJDUMP-NEXT: 0000000000000001 l       *ABS*  0000000000000000 min_recursive_subexpr
// OBJDUMP-NEXT: 7fffffffffffffff l       *ABS*  0000000000000000 min_expr_one_max
// OBJDUMP-NEXT: 0000000000000003 l       *ABS*  0000000000000000 min_expr_two_max
// OBJDUMP-NEXT: ffffffffff676980 l       *ABS*  0000000000000000 min_expr_three_max
// OBJDUMP-NEXT: 8000000000000000 l       *ABS*  0000000000000000 min_expr_one_min
// OBJDUMP-NEXT: 8000000000000000 l       *ABS*  0000000000000000 min_expr_two_min
// OBJDUMP-NEXT: 8000000000000000 l       *ABS*  0000000000000000 min_expr_three_min
// OBJDUMP-NEXT: 0000000000000007 l       *ABS*  0000000000000000 or_expression_all
// OBJDUMP-NEXT: 0000000000000003 l       *ABS*  0000000000000000 or_expression_two
// OBJDUMP-NEXT: 0000000000000001 l       *ABS*  0000000000000000 or_expression_one
// OBJDUMP-NEXT: 000000000000000f l       *ABS*  0000000000000000 or_literals
// OBJDUMP-NEXT: 0000000000000000 l       *ABS*  0000000000000000 or_false
// OBJDUMP-NEXT: 00000000000000ff l       *ABS*  0000000000000000 or_with_or_sym
// OBJDUMP-NEXT: 00000000000000ff l       *ABS*  0000000000000000 or
// OBJDUMP-NEXT: 0000000000000003 l       *ABS*  0000000000000000 or_with_subexpr
// OBJDUMP-NEXT: 0000000000000008 l       *ABS*  0000000000000000 or_as_subexpr
// OBJDUMP-NEXT: 0000000000000007 l       *ABS*  0000000000000000 or_recursive_subexpr

// ASM: .set zero, 0
// ASM: .set one, 1
// ASM: .set two, 2
// ASM: .set three, 3
// ASM: .set i64_max, 9223372036854775807
// ASM: .set i64_min, -9223372036854775808

.set zero, 0
.set one, 1
.set two, 2
.set three, 3
.set i64_max, 0x7FFFFFFFFFFFFFFF
.set i64_min, 0x8000000000000000

// ASM: .set max_expression_all, max(1, 2, five, 3, four)
// ASM: .set max_expression_two, 2
// ASM: .set max_expression_one, 1
// ASM: .set max_literals, 10
// ASM: .set max_with_max_sym, max(max, 4, 3, 1, 2)

.set max_expression_all, max(one, two, five, three, four)
.set max_expression_two, max(one, two)
.set max_expression_one, max(one)
.set max_literals, max(1,2,3,4,5,6,7,8,9,10)
.set max_with_max_sym, max(max, 4, 3, one, two)

// ASM: .set max_neg_numbers, -1
// ASM: .set max_neg_number, -1

.set neg_one, -1
.set max_neg_numbers, max(-5, -4, -3, -2, neg_one)
.set max_neg_number, max(neg_one)

// ASM: .set max_with_subexpr, 3
// ASM: .set max_as_subexpr, 1+max(4, 3, five)
// ASM: .set max_recursive_subexpr, max(max(1, four), 3, max_expression_all)

.set max_with_subexpr, max(((one | 3) << 3) / 8)
.set max_as_subexpr, 1 + max(4, 3, five)
.set max_recursive_subexpr, max(max(one, four), three, max_expression_all)

// ASM: .set max_expr_one_max, 9223372036854775807
// ASM: .set max_expr_two_max, max(9223372036854775807, five)
// ASM: .set max_expr_three_max, max(9223372036854775807, five, 10000000)

.set max_expr_one_max, max(i64_max)
.set max_expr_two_max, max(i64_max, five)
.set max_expr_three_max, max(i64_max, five, 10000000)

// ASM: .set max_expr_one_min, -9223372036854775808
// ASM: .set max_expr_two_min, 3
// ASM: .set max_expr_three_min, 10000000

.set max_expr_one_min, max(i64_min)
.set max_expr_two_min, max(i64_min, three)
.set max_expr_three_min, max(i64_min, three, 10000000)

// ASM: .set min_expression_all, min(1, 2, five, 3, four)
// ASM: .set min_expression_two, 1
// ASM: .set min_expression_one, 3
// ASM: .set min_literals, 1
// ASM: .set min_with_min_sym, min(min, 4, 3, 1, 2)

.set min_expression_all, min(one, two, five, three, four)
.set min_expression_two, min(one, three)
.set min_expression_one, min(three)
.set min_literals, min(1,2,3,4,5,6,7,8,9,10)
.set min_with_min_sym, min(min, 4, 3, one, two)

// ASM: .set min_neg_numbers, -5
// ASM: .set min_neg_number, -1

.set neg_one, -1
.set min_neg_numbers, min(-5, -4, -3, -2, neg_one)
.set min_neg_number, min(neg_one)

// ASM: .set min_with_subexpr, 3
// ASM: .set min_as_subexpr, 1+min(4, 3, five)
// ASM: .set min_recursive_subexpr, min(min(1, four), 3, min_expression_all)

.set min_with_subexpr, min(((one | 3) << 3) / 8)
.set min_as_subexpr, 1 + min(4, 3, five)
.set min_recursive_subexpr, min(min(one, four), three, min_expression_all)

// ASM: .set min_expr_one_max, 9223372036854775807
// ASM: .set min_expr_two_max, 3
// ASM: .set min_expr_three_max, -10000000

.set min_expr_one_max, min(i64_max)
.set min_expr_two_max, min(i64_max, three)
.set min_expr_three_max, min(i64_max, three, -10000000)

// ASM: .set min_expr_one_min, -9223372036854775808
// ASM: .set min_expr_two_min, min(-9223372036854775808, five)
// ASM: .set min_expr_three_min, min(-9223372036854775808, five, 10000000)

.set min_expr_one_min, min(i64_min)
.set min_expr_two_min, min(i64_min, five)
.set min_expr_three_min, min(i64_min, five, 10000000)

// ASM: .set or_expression_all, or(1, 2, five, 3, four)
// ASM: .set or_expression_two, 3
// ASM: .set or_expression_one, 1
// ASM: .set or_literals, 15
// ASM: .set or_false, 0
// ASM: .set or_with_or_sym, or(or, 4, 3, 1, 2)

.set or_expression_all, or(one, two, five, three, four)
.set or_expression_two, or(one, two)
.set or_expression_one, or(one)
.set or_literals, or(1,2,3,4,5,6,7,8,9,10)
.set or_false, or(zero, 0, (2-2), 5 > 6)
.set or_with_or_sym, or(or, 4, 3, one, two)

// ASM: .set or_with_subexpr, 3
// ASM: .set or_as_subexpr, 1+or(4, 3, five)
// ASM: .set or_recursive_subexpr, or(or(1, four), 3, or_expression_all)

.set or_with_subexpr, or(((one | 3) << 3) / 8)
.set or_as_subexpr, 1 + or(4, 3, five)
.set or_recursive_subexpr, or(or(one, four), three, or_expression_all)

// ASM: .set four, 4
// ASM: .set five, 5
// ASM: .set max, 15
// ASM: .set or, 255

.set four, 4
.set five, 5
.set max, 0xF
.set min, 0x0
.set or, 0xFF
