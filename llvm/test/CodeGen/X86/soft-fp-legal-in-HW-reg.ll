; RUN: llc < %s -mtriple=x86_64-linux-android -mattr=+mmx -enable-legalize-types-checking | FileCheck %s
;
; D31946
; Check that we dont end up with the ""LLVM ERROR: Cannot select" error.
; Additionally ensure that the output code actually put fp128 values in SSE registers.

declare fp128 @llvm.fabs.f128(fp128)
declare fp128 @llvm.copysign.f128(fp128, fp128)

define fp128 @TestSelect(fp128 %a, fp128 %b) {
  %cmp = fcmp ogt fp128 %a, %b
  %sub = fsub fp128 %a, %b
  %res = select i1 %cmp, fp128 %sub, fp128 0xL00000000000000000000000000000000
  ret fp128 %res
; CHECK-LABEL: TestSelect:
; CHECK        movaps 16(%rsp), %xmm1
; CHECK-NEXT   callq __subtf3
; CHECK-NEXT   testl %ebx, %ebx
; CHECK-NEXT   jg .LBB0_2
; CHECK-NEXT # %bb.1:
; CHECK-NEXT   movaps .LCPI0_0(%rip), %xmm0
; CHECK-NEXT .LBB0_2:
; CHECK-NEXT   addq $32, %rsp
; CHECK-NEXT   popq %rbx
; CHECK-NEXT   retq
}

define fp128 @TestFabs(fp128 %a) {
  %res = call fp128 @llvm.fabs.f128(fp128 %a)
  ret fp128 %res
; CHECK-LABEL: TestFabs:
; CHECK      andps .LCPI1_0(%rip), %xmm0
; CHECK-NEXT retq
}

define fp128 @TestCopysign(fp128 %a, fp128 %b) {
  %res = call fp128 @llvm.copysign.f128(fp128 %a, fp128 %b)
  ret fp128 %res
; CHECK-LABEL: TestCopysign:
; CHECK      andps .LCPI2_1(%rip), %xmm0
; CHECK-NEXT orps %xmm1, %xmm0
; CHECK-NEXT retq
}

define fp128 @TestFneg(fp128 %a) {
  %mul = fmul fp128 %a, %a
  %res = fsub fp128 0xL00000000000000008000000000000000, %mul
  ret fp128 %res
; CHECK-LABEL: TestFneg:
; CHECK      movaps %xmm0, %xmm1
; CHECK-NEXT callq __multf3
; CHECK-NEXT xorps .LCPI3_0(%rip), %xmm0
; CHECK-NEXT popq %rax
; CHECK-NEXT retq
}
