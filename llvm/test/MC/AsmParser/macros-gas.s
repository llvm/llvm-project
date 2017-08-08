// RUN: not llvm-mc -triple i386-linux-gnu %s 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err

.macro .test0
.macrobody0
.endm
.macro .test1
.test0
.endm

.test1
// CHECK-ERRORS: <instantiation>:1:1: error: unknown directive
// CHECK-ERRORS-NEXT: macrobody0
// CHECK-ERRORS-NEXT: ^
// CHECK-ERRORS: <instantiation>:1:1: note: while in macro instantiation
// CHECK-ERRORS-NEXT: .test0
// CHECK-ERRORS-NEXT: ^
// CHECK-ERRORS: 11:1: note: while in macro instantiation
// CHECK-ERRORS-NEXT: .test1
// CHECK-ERRORS-NEXT: ^

.macro test2 _a
.byte \_a
.endm
// CHECK: .byte 10
test2 10

.macro test3 _a _b _c
.ascii "\_a \_b \_c \\_c"
.endm

// CHECK: .ascii "1 2 3 \003"
test3 1, 2, 3

// CHECK: .ascii "1 2 3 \003"
test3 1, 2 3

.macro test3_prime _a _b _c
.ascii "\_a \_b \_c"
.endm

// CHECK: .ascii "1 (2 3) "
test3_prime 1, (2 3)

// CHECK: .ascii "1 (2 3) "
test3_prime 1 (2 3)

// CHECK: .ascii "1 2 "
test3_prime 1 2

.macro test5 _a
.globl \_a
.endm

// CHECK: .globl zed1
test5 zed1

.macro test6 $a
.globl \$a
.endm

// CHECK: .globl zed2
test6 zed2

.macro test7 .a
.globl \.a
.endm

// CHECK: .globl zed3
test7 zed3

.macro test8 _a, _b, _c
.ascii "\_a,\_b,\_c"
.endm

.macro test9 _a _b _c
.ascii "\_a \_b \_c"
.endm

// CHECK: .ascii "a,b,c"
test8 a, b, c
// CHECK: .ascii "%1,%2,%3"
test8 %1 %2 %3 #a comment
// CHECK: .ascii "x-y,z,1"
test8 x - y z 1
// CHECK: .ascii "1 2 3"
test9 1, 2,3

// CHECK: .ascii "1,2,3"
test8 1,2 3

// CHECK: .ascii "1,2,3"
test8 1 2, 3

.macro test10
.ascii "$20"
.endm

test10
// CHECK: .ascii "$20"

test10 42
// CHECK-ERRORS: 102:10: error: Wrong number of arguments
// CHECK-ERRORS-NEXT: test10 42
// CHECK-ERRORS-NEXT: ^
