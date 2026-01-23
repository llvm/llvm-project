; RUN: opt < %s -mtriple=s390x-unknown-linux-gnu -mcpu=z16 -S -passes=infer-alignment \
; RUN:   2>&1 | FileCheck %s
;
; Test that the alignment of the alloca is not increased beyond the stack
; alignment of 8 bytes.

declare void @foo(ptr)

define void @f1(<4 x i64> %Arg) {
; CHECK-LABEL: define void @f1
; CHECK-NEXT:    %param = alloca <4 x i64>, align 8
; CHECK-NEXT:    store <4 x i64> %Arg, ptr %param, align 8
  %param = alloca <4 x i64>, align 8
  store <4 x i64> %Arg, ptr %param, align 8
  call void @foo(ptr %param)
  ret void
}
