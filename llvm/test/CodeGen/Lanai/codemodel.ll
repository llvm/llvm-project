; RUN: llc -mtriple=lanai < %s | FileCheck %s
; RUN: llc -mtriple=lanai < %s -code-model=small  | FileCheck -check-prefix CHECK-SMALL %s
; RUN: not llc -mtriple=lanai < %s -code-model=tiny 2>&1 | FileCheck -check-prefix CHECK-TINY %s
; RUN: not llc -mtriple=lanai < %s -code-model=kernel 2>&1 | FileCheck -check-prefix CHECK-KERNEL %s

; CHECK-TINY: Target does not support the tiny CodeModel
; CHECK-KERNEL: Target does not support the kernel CodeModel

@data = external global [0 x i32]		; <[0 x i32]*> [#uses=5]

define i32 @foo() nounwind readonly {
entry:
; CHECK-SMALL-LABEL:  foo:
; CHECK-SMALL: ld [data], %rv
; CHECK-LABEL:  foo:
; CHECK: mov hi(data), %r[[REGISTER:[0-9]+]]
; CHECK: or %r[[REGISTER]], lo(data), %r[[REGISTER]]
; CHECK: ld 0[%r[[REGISTER]]], %rv
	%0 = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @data, i64 0, i64 0), align 4		; <i32> [#uses=1]
	ret i32 %0
}

define i32 @foo1() nounwind readonly {
entry:
; CHECK-SMALL-LABEL:  foo1:
; CHECK-SMALL: mov data, %r[[REGISTER:[0-9]+]]
; CHECK-SMALL: ld 40[%r[[REGISTER]]], %rv
; CHECK-LABEL:  foo1:
; CHECK: mov hi(data), %r[[REGISTER:[0-9]+]]
; CHECK: or %r[[REGISTER]], lo(data), %r[[REGISTER]]
; CHECK: ld 40[%r[[REGISTER]]], %rv
	%0 = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @data, i32 0, i64 10), align 4		; <i32> [#uses=1]
	ret i32 %0
}

@y = local_unnamed_addr global i32* null, section ".ldata,block", align 8

define i32 @foo2() nounwind readonly {
entry:
; CHECK-SMALL-LABEL:  foo2:
; CHECK-SMALL: mov hi(y), %r[[REGISTER:[0-9]+]]
; CHECK-SMALL: or %r[[REGISTER]], lo(y), %r[[REGISTER]]
; CHECK-LABEL:  foo2:
; CHECK: mov hi(y), %r[[REGISTER:[0-9]+]]
; CHECK: or %r[[REGISTER]], lo(y), %r[[REGISTER]]
  %0 = load i32*, i32** @y, align 8
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}
