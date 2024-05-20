# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s

.macro empty, cond
.endm
empty ne

# CHECK: .ascii "3 \003"
.macro escape a
.ascii "\a \\a"
.endm
escape 3

.macro double first = -1, second = -1
  .long \first
  .long \second
.endm

# CHECK:      .long -1
# CHECK-NEXT: .long -1
# CHECK-EMPTY:
double
# CHECK:      .long -1
# CHECK-NEXT: .long -1
# CHECK-EMPTY:
double ,
# CHECK:      .long 1
# CHECK-NEXT: .long -1
double 1
# CHECK:      .long 2
# CHECK-NEXT: .long 3
double 2, 3
# CHECK:      .long -1
# CHECK-NEXT: .long 4
double , 4
# CHECK:      .long 5
# CHECK-NEXT: .long 6
double 5, second = 6
# CHECK:      .long 7
# CHECK-NEXT: .long -1
double first = 7
# CHECK:      .long -1
# CHECK-NEXT: .long 8
double second = 8
# CHECK:      .long 10
# CHECK-NEXT: .long 9
double second = 9, first = 10
# CHECK:      .long second+11
# CHECK-NEXT: .long -1
double second + 11
# CHECK:      .long -1
# CHECK-NEXT: .long second+12
double , second + 12
# CHECK:      .long second
# CHECK-NEXT: .long -1
double second

.macro mixed arg0 = 0, arg1 = 1 arg2 = 2, arg3 = 3
  .long \arg0
  .long \arg1
  .long \arg2
  .long \arg3
.endm

# CHECK:      .long 1
# CHECK-NEXT: .long 2
# CHECK-NEXT: .long 3
# CHECK-NEXT: .long 3
mixed 1, 2 3

# CHECK:      .long 1
# CHECK-NEXT: .long 2
# CHECK-NEXT: .long 3
# CHECK-NEXT: .long 3
mixed 1 2, 3

# CHECK:      .long 1
# CHECK-NEXT: .long 2
# CHECK-NEXT: .long 3
# CHECK-NEXT: .long 4
mixed 1 2, 3 4

.macro ascii3 _a _b _c
.ascii "\_a|\_b|\_c"
.endm

## 3 arguments.
# CHECK: .ascii "a|b|c"
ascii3 a, b, c
# CHECK: .ascii "%1|%2|%3"
ascii3 %1 %2 %3
# CHECK: .ascii "1|2|3"
ascii3 1, 2,3
# CHECK: .ascii "1|2|3"
ascii3 1,2 3
# CHECK: .ascii "1|2|3"
ascii3 1 2, 3
# CHECK: .ascii "x-y|z|1"
ascii3 x - y z 1

## 2 arguments.
# CHECK: .ascii "1|(2 3)|"
ascii3 1, (2 3)
# CHECK: .ascii "1|(2 3)|"
ascii3 1 (2 3)
