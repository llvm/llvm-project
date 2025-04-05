; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=attributes --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT %s < %t

; Test that convergent attributes are reduced if convergencectrl
; bundles are not used

; INTERESTING-LABEL: define float @convergent_intrinsic(
; RESULT-LABEL: define float @convergent_intrinsic(float %x, float %y) {
define float @convergent_intrinsic(float %x, float %y) #0 {
  %val = call float @llvm.amdgcn.readfirstlane.f32(float %x)
  ret float %val
}

; INTERESTING-LABEL: define float @convergent_func_no_convergent_intrinsics(
; RESULT-LABEL: define float @convergent_func_no_convergent_intrinsics(float %x, float %y) {
define float @convergent_func_no_convergent_intrinsics(float %x, float %y) #0 {
  %val = call float @extern.func(float %x, float %y)
  ret float %val
}


; RESULT-LABEL: declare float @convergent.extern.func(float, float){{$}}
declare float @convergent.extern.func(float, float) #0
declare float @extern.func(float, float)
declare float @llvm.amdgcn.readfirstlane.f32(float) #1

; RESULT: attributes #0 = { convergent nocallback nofree nounwind willreturn memory(none) }
; RESULT-NOT: attributes

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nocallback nofree nounwind willreturn memory(none) }
