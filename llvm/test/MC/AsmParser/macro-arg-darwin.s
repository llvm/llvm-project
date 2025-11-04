// RUN: llvm-mc -triple x86_64-apple-darwin10 %s | FileCheck %s

.macro GET   var,re2g
    movl   \var@GOTOFF(%ebx),\re2g
.endm

.macro GET_DEFAULT var, re2g=%ebx, re3g=%ecx
movl 2(\re2g, \re3g, 2), \var
.endm

GET         is_sse, %eax
// CHECK: movl  is_sse@GOTOFF(%ebx), %eax

GET_DEFAULT %ebx, , %edx
// CHECK: movl  2(%ebx,%edx,2), %ebx

GET_DEFAULT %ebx, %edx
// CHECK: movl  2(%edx,%ecx,2), %ebx

.macro bar
    .long $n
.endm

bar 1, 2, 3
bar

// CHECK: .long 3
// CHECK: .long 0


.macro top
    middle _$0, $1
.endm
.macro middle
     $0:
    .if $n > 1
        bottom $1
    .endif
.endm
.macro bottom
    fred = $0
.endm

.text

top foo
top bar, 42

// CHECK: _foo:
// CHECK-NOT: fred
// CHECK: _bar
// CHECK-NEXT: fred = 42


.macro foo
foo_$0_$1_$2_$3:
  nop
.endm

foo 1, 2, 3, 4
foo 1, , 3, 4

// CHECK: foo_1_2_3_4:
// CHECK: foo_1__3_4:
