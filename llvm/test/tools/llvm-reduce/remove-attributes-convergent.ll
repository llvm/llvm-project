; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=attributes --test FileCheck --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,RESULT %s < %t

; Test that invalid reductions aren't produced on convergent functions
; which use convergencectrl bundles

; CHECK-LABEL: define float @convergent_intrinsic(float %x, float %y)
; RESULT-SAME: [[CONVERGENT_ONLY:#[0-9]+]] {
define float @convergent_intrinsic(float %x, float %y) #0 {
  %entry.token = call token @llvm.experimental.convergence.entry()
  %val = call float @llvm.amdgcn.readfirstlane.f32(float %x) [ "convergencectrl"(token %entry.token) ]
  ret float %val
}

; CHECK-LABEL: define float @convergent_callsite(float %x, float %y)
; RESULT-SAME: [[CONVERGENT_ONLY]] {
; RESULT: call float @extern.func(float %x, float %y) [[CONVERGENT_ONLY]]
define float @convergent_callsite(float %x, float %y) #0 {
  %entry.token = call token @llvm.experimental.convergence.entry()
  %val = call float @extern.func(float %x, float %y) #0 [ "convergencectrl"(token %entry.token) ]
  ret float %val
}

; CHECK-LABEL: define float @convergent_declaration(float %x, float %y)
; RESULT-SAME: [[CONVERGENT_ONLY]] {
define float @convergent_declaration(float %x, float %y) #0 {
  %entry.token = call token @llvm.experimental.convergence.entry()
  %val = call float @convergent.extern.func(float %x, float %y) [ "convergencectrl"(token %entry.token) ]
  ret float %val
}

; CHECK-LABEL: define float @convergent_func_no_convergent_intrinsics(float %x, float %y)
; RESULT-SAME: [[CONVERGENT_ONLY]] {
define float @convergent_func_no_convergent_intrinsics(float %x, float %y) #0 {
  %val = call float @extern.func(float %x, float %y)
  ret float %val
}

; CHECK-LABEL: define float @convergent_func_no_convergent_bundles(float %x, float %y)
; RESULT-SAME: [[CONVERGENT_ONLY]] {
define float @convergent_func_no_convergent_bundles(float %x, float %y) #0 {
  %entry.token = call token @llvm.experimental.convergence.entry()
  %val = call float @extern.func(float %x, float %y)
  ret float %val
}

; CHECK-LABEL: declare float @convergent.extern.func(float, float)
; RESULT-SAME: [[CONVERGENT_ONLY]]{{$}}
declare float @convergent.extern.func(float, float) #0
declare float @extern.func(float, float)
declare float @llvm.amdgcn.readfirstlane.f32(float) #1
declare token @llvm.experimental.convergence.entry() #2

; RESULT: attributes [[CONVERGENT_ONLY]] = { convergent }

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
