// RUN: not llvm-mc -triple i386-apple-darwin10 %s 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err

.macro .test0
.macrobody0
.endmacro
.macro .test1
.test0
.endmacro

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

.macro test2
.byte $0
.endmacro
// CHECK: .byte 10
test2 10

.macro test3
.globl "$0 $1 $2 $$3 $n"
.endmacro

// CHECK: .globl "1 23  $3 2"
test3 1, 2 3

// CHECK: .globl	"1 (23)  $3 2"
test3 1, (2 3)

// CHECK: .globl "12  $3 1"
test3 1 2

.macro test4
.globl "$0 -- $1"
.endmacro

// CHECK: .globl  "(ab)(,)) -- (cd)"
test4 (a b)(,)),(cd)

// CHECK: .globl  "(ab)(,)) -- (cd)"
test4 (a b)(,)),(cd)

.macro test5 _a
.globl "\_a"
.endm

// CHECK: .globl zed1
test5 zed1

.macro test6 $a
.globl "\$a"
.endm

// CHECK: .globl zed2
test6 zed2

.macro test7 .a
.globl "\.a"
.endm

// CHECK: .globl zed3
test7 zed3

.macro test8 _a, _b, _c
.globl "\_a,\_b,\_c"
.endmacro

.macro test9 _a _b _c
.globl "\_a \_b \_c"
.endmacro

// CHECK: .globl  "a,b,c"
test8 a, b, c
// CHECK: .globl  "%1,%2,%3"
test8 %1, %2, %3 #a comment
// CHECK: .globl "x-y,z,1"
test8 x - y, z, 1
// CHECK: .globl  "1 2 3"
test9 1, 2,3

// CHECK: .globl "1,23,"
test8 1,2 3

// CHECK: .globl "12,3,"
test8 1 2, 3
