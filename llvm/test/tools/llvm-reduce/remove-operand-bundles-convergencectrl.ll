; Check that invalid reductions aren't introduced by deleting
; convergencectrl bundles in convergent functions
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operand-bundles --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,RESULT %s < %t

; CHECK-LABEL: define float @convergentctrl_one_interesting(
; INTERESTING: %interesting = call float @convergent.extern.func(
; RESULT: %entry.token = call token @llvm.experimental.convergence.entry()
; RESULT: %interesting = call float @convergent.extern.func(float %x) [ "convergencectrl"(token %entry.token) ]
; RESULT: %boring = call float @convergent.extern.func(float %x) [ "convergencectrl"(token %entry.token) ]
define float @convergentctrl_one_interesting(float %x, float %y) #0 {
  %entry.token = call token @llvm.experimental.convergence.entry()
  %interesting = call float @convergent.extern.func(float %x) [ "convergencectrl"(token %entry.token) ]
  %boring = call float @convergent.extern.func(float %x) [ "convergencectrl"(token %entry.token) ]
  %add = fadd float %interesting, %boring
  ret float %add
}

; In theory we could remove the bundle here, since all convergencectrl
; in the function will be removed.

; CHECK-LABEL: define float @convergentctrl_can_remove_all(
; RESULT: %entry.token = call token @llvm.experimental.convergence.entry()
; RESULT: %val = call float @convergent.extern.func(float %x) [ "convergencectrl"(token %entry.token) ]
define float @convergentctrl_can_remove_all(float %x, float %y) #0 {
  %entry.token = call token @llvm.experimental.convergence.entry()
  %val = call float @convergent.extern.func(float %x) [ "convergencectrl"(token %entry.token) ]
  ret float %val
}

declare float @convergent.extern.func(float) #0
declare token @llvm.experimental.convergence.entry() #1

attributes #0 = { convergent }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
