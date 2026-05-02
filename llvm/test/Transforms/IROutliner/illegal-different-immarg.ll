; RUN: opt -S -passes=verify,iroutliner -ir-outlining-no-cost < %s | FileCheck %s

; Do not outline structurally similar regions when doing so would replace a
; non-uniform immarg constant with an outlined-function argument.

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

define ptr @PR194733() {
entry:
  br i1 false, label %then, label %else

then:
  %0 = load ptr, ptr null, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr null, ptr null, i64 0, i1 false)
  ret ptr null

else:
  %1 = load ptr, ptr null, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr null, ptr null, i64 0, i1 true)
  %2 = load ptr, ptr null, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr null, ptr null, i64 0, i1 false)
  ret ptr null
}
; CHECK-LABEL: define ptr @PR194733(
; CHECK-NOT: outlined_ir_func
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr null, ptr null, i64 0, i1 false)
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr null, ptr null, i64 0, i1 true)
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr null, ptr null, i64 0, i1 false)
