; RUN: opt -S -gvn-hoist < %s | FileCheck %s

; Check that the call and fcmp are hoisted.
; CHECK-LABEL: define void @fun(
; CHECK: call float
; CHECK: fcmp oeq
; CHECK-NOT: call float
; CHECK-NOT: fcmp oeq

define void @fun(float %__b) minsize {
entry:
  br label %if.then

if.then:                                          ; preds = %entry
  br i1 undef, label %if.then8, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.then
  %0 = call float @llvm.fabs.f32(float %__b) #2
  %cmpinf7 = fcmp oeq float %0, 0x7FF0000000000000
  unreachable

if.then8:                                         ; preds = %if.then
  %1 = call float @llvm.fabs.f32(float %__b) #2
  %cmpinf10 = fcmp oeq float %1, 0x7FF0000000000000
  ret void
}

declare float @llvm.fabs.f32(float)
