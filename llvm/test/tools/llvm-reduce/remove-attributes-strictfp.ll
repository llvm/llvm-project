; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=attributes --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,RESULT %s < %t

; Test that invalid reductions aren't produced on strictfp functions.

; CHECK-LABEL: define float @strictfp_intrinsic(float %x, float %y)
; RESULT-SAME: [[STRICTFP_ONLY:#[0-9]+]] {
define float @strictfp_intrinsic(float %x, float %y) #0 {
  %val = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %val
}

; CHECK-LABEL: define float @strictfp_callsite(float %x, float %y)
; RESULT-SAME: [[STRICTFP_ONLY]] {
; RESULT: call float @extern.func(float %x, float %y) [[STRICTFP_ONLY]]
define float @strictfp_callsite(float %x, float %y) #0 {
  %val = call float @extern.func(float %x, float %y) #0
  ret float %val
}

; CHECK-LABEL: define float @strictfp_declaration(float %x, float %y)
; RESULT-SAME: [[STRICTFP_ONLY]] {
define float @strictfp_declaration(float %x, float %y) #0 {
  %val = call float @strict.extern.func(float %x, float %y)
  ret float %val
}

; CHECK-LABEL: define float @strictfp_no_constrained_ops(float %x, float %y)
; RESULT-SAME: [[STRICTFP_ONLY]] {
define float @strictfp_no_constrained_ops(float %x, float %y) #0 {
  %val = call float @llvm.copysign.f32(float %x, float %y) #1
  ret float %val
}

; CHECK-LABEL: declare float @strict.extern.func(float, float)
; RESULT-SAME: [[STRICTFP_ONLY]]{{$}}
declare float @strict.extern.func(float, float) #0

declare float @extern.func(float, float)

declare float @llvm.copysign.f32(float, float)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)

; RESULT: attributes [[STRICTFP_ONLY]] = { strictfp }

attributes #0 = { nounwind strictfp }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) strictfp }
