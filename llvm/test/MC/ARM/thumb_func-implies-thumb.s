@ RUN: llvm-mc -triple=armv7-darwin- -show-encoding < %s | FileCheck %s
.syntax unified

.text

.arm
@ Ensure the plain form switches mode.
.thumb_func
@ CHECK: .code 16
@ CHECK-LABEL: foo
foo:
    bx lr

.arm
@ Ensure the labeled form doesn't switch mode.
.thumb_func bar
@ CHECK-NOT: .code 16
@ CHECK-LABEL: bar
bar:
    bx lr

.arm
@ Ensure the nop is assembled in thumb mode, even though the baz symbol is
@ defined later.
.thumb_func
nop
@ CHECK: .code 16
@ CHECK-NEXT: nop
@ CHECK-LABEL: baz
baz:
    bx lr
