; RUN: llc %s -mattr=+avx -o - | FileCheck %s
; PR21743.

target triple = "x86_64-pc-win32-elf"

; Check that copy propagation conservatively assumes that undef register
; can be rewritten by the backend to break false dependencies for the
; hardware.
; In this function we are in this situation:
; reg1 = copy reg2
; = inst reg2<undef>
; reg2 = copy reg1
; Copy propagation used to remove the last copy.
; This is incorrect because the undef flag on reg2 in inst, allows next
; passes to put whatever trashed value in reg2 that may help.
; In practice we end up with this code:
; reg1 = copy reg2
; reg2 = 0
; = inst reg2<undef>
; reg2 = copy reg1
; Therefore, removing the last copy is wrong.
;
; CHECK-LABEL: foo:
; CHECK: movl	$339752784, %e[[INDIRECT_CALL1:[a-z]+]]
; CHECK: callq *%r[[INDIRECT_CALL1]]
; Copy the result in a temporary.
; Note: Technically the regalloc could have been smarter and this move not required,
; which would have hidden the bug.
; CHECK: vmovapd	%xmm0, [[TMP:%xmm[0-9]+]]
; Crush xmm0.
; CHECK-NEXT: vxorps %xmm0, %xmm0, %xmm0
; CHECK: movl	$339772768, %e[[INDIRECT_CALL2:[a-z]+]]
; Set TMP in the first argument of the second call.
; CHECK-NEXT: vmovapd	[[TMP]], %xmm0
; CHECK: callq *%r[[INDIRECT_CALL2]]
; CHECK: retq
define double @foo(i64 %arg) {
top:
  %tmp = call double inttoptr (i64 339752784 to double (double, double)*)(double 1.000000e+00, double 0.000000e+00)
  tail call void asm sideeffect "", "x,~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{dirflag},~{fpsr},~{flags}"(double %tmp)
  %tmp1 = sitofp i64 %arg to double
  call void inttoptr (i64 339772768 to void (double, double)*)(double %tmp, double %tmp1)
  %tmp3 = fadd double %tmp1, %tmp
  ret double %tmp3
}
