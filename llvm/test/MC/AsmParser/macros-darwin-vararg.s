// RUN: llvm-mc -triple i386-apple-darwin10 %s 2>&1 | FileCheck %s

.macro abc a b:vararg
.globl "\a, \b"
.endm

// CHECK: .globl "zed0, zed1, zed2"
abc zed0, zed1, zed2

.purgem abc

.macro ifcc arg:vararg
.if cc
            \arg
.endif
.endm

.macro ifcc2 arg0 arg1:vararg
.if cc
            movl \arg0, \arg1
.endif
.endm

.macro ifcc3 arg0, arg1:vararg
.if cc
            movl \arg0, \arg1
.endif
.endm

.macro ifcc4 arg0, arg1:vararg
.if cc
            movl \arg1, \arg0
.endif
.endm

.text

// CHECK: movl %esp, %ebp
// CHECK: subl $0, %esp
// CHECK: movl %eax, %ebx
// CHECK: movl %ecx, %ebx
// CHECK: movl %ecx, %eax
// CHECK: movl %eax, %ecx
// CHECK: movl %ecx, %eax
// CHECK: movl %eax, %ecx
.set cc,1
  ifcc  movl    %esp, %ebp
        subl $0, %esp

  ifcc2 %eax, %ebx
  ifcc2 %ecx, %ebx
  ifcc3 %ecx, %eax
  ifcc3 %eax, %ecx
  ifcc4 %eax, %ecx  ## test
  ifcc4 %ecx, %eax ## test

// CHECK-NOT: movl
// CHECK: subl $1, %esp
.set cc,0
  ifcc  movl,    %esp, %ebp
        subl $1, %esp

.macro abc arg:vararg=nop
  \arg
.endm

.macro abcd arg0=%eax, arg1:vararg=%ebx
  movl \arg0, \arg1
.endm

.text

// CHECK: nop
  abc
// CHECK: movl %eax, %ebx
  abcd ,

.macro .make_macro start, end, name, body:vararg
\start \name
\body
\end
.endmacro

.make_macro .macro,.endmacro,.mybyte,.byte $0, $2, $1

.data
// CHECK: .byte 10
// CHECK: .byte 12
// CHECK: .byte 11
.mybyte 10,11,12
