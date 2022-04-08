# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

## Test that a variable declared with "var = other_var + cst" is in the same
## section as other_var and its value is the value of other_var + cst.
## In addition, its st_size inherits from other_var.

# CHECK:      0: {{.*}}
# CHECK-NEXT:    0000000000000001    42 OBJECT  GLOBAL DEFAULT [[#A:]] a
# CHECK-NEXT:    0000000000000005     0 NOTYPE  GLOBAL DEFAULT [[#A]]  b
# CHECK-NEXT:    0000000000000001    42 OBJECT  GLOBAL DEFAULT [[#A]]  a1
# CHECK-NEXT:    0000000000000002    42 OBJECT  GLOBAL DEFAULT [[#A]]  c
# CHECK-NEXT:    000000000000000d    42 OBJECT  GLOBAL DEFAULT [[#A]]  d
# CHECK-NEXT:    000000000000000d    42 OBJECT  GLOBAL DEFAULT [[#A]]  d1
# CHECK-NEXT:    000000000000000d    42 OBJECT  GLOBAL DEFAULT [[#A]]  d2
# CHECK-NEXT:    0000000000000001    41 OBJECT  GLOBAL DEFAULT [[#A]]  e
# CHECK-NEXT:    0000000000000001    42 OBJECT  GLOBAL DEFAULT [[#A]]  e1
# CHECK-NEXT:    0000000000000001    42 OBJECT  GLOBAL DEFAULT [[#A]]  e2
# CHECK-NEXT:    0000000000000002    42 OBJECT  GLOBAL DEFAULT [[#A]]  e3
# CHECK-NEXT:    0000000000000005     0 NOTYPE  GLOBAL DEFAULT [[#A]]  test2_a
# CHECK-NEXT:    0000000000000005     0 NOTYPE  GLOBAL DEFAULT [[#A]]  test2_b
# CHECK-NEXT:    0000000000000009     0 NOTYPE  GLOBAL DEFAULT [[#A]]  test2_c
# CHECK-NEXT:    0000000000000009     0 NOTYPE  GLOBAL DEFAULT [[#A]]  test2_d
# CHECK-NEXT:    0000000000000004     0 NOTYPE  GLOBAL DEFAULT  ABS    test2_e
# CHECK-NEXT:    0000000000000001    42 OBJECT  GLOBAL DEFAULT [[#A]]  e@v1


        .data
        .globl	a
        .size a, 42
        .byte 42
        .type a, @object
a:

        .long 42
        .globl b, a1, c, d, d1, d2, e, e1, e2, e3
b:
a1 = a
c = a + 1

## These st_size fields inherit from a.
d = a + (b - a) * 3
.set d1, d
d2 = d1

e = a + (1 - 1)
.size e, 41
## FIXME These st_size fields inherit from e instead of a.
.set e1, e
.set e2, e1
e3 = e1 + 1

        .globl test2_a
        .globl test2_b
        .globl test2_c
        .globl test2_d
        .globl test2_e
test2_a:
    .long 0
test2_b = test2_a
test2_c:
    .long 0
test2_d = test2_c
test2_e = test2_d - test2_b

## e@v1's st_size equals e's st_size.
.symver e, e@v1
