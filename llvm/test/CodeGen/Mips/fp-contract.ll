; Test that fmul and fadd are fused into fmadd only when the contract fast-math
; flag is present on the operations (likewise fmul, fsub and fmsub).

; RUN: llc -mtriple=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

declare <4 x float> @llvm.mips.fmul.w(<4 x float>, <4 x float>)
declare <4 x float> @llvm.mips.fadd.w(<4 x float>, <4 x float>)
declare <4 x float> @llvm.mips.fsub.w(<4 x float>, <4 x float>)

; Without the contract flag, fmul and fadd stay separate.
define void @foo(ptr %agg.result, ptr %acc, ptr %a, ptr %b) {
entry:
  %0 = load <4 x float>, ptr %a, align 16
  %1 = load <4 x float>, ptr %b, align 16
  %2 = call <4 x float> @llvm.mips.fmul.w(<4 x float> %0, <4 x float> %1)
  %3 = load <4 x float>, ptr %acc, align 16
  %4 = call <4 x float> @llvm.mips.fadd.w(<4 x float> %3, <4 x float> %2)
  store <4 x float> %4, ptr %agg.result, align 16
  ret void
  ; CHECK-LABEL: foo:
  ; CHECK: fmul.w
  ; CHECK: fadd.w
}

; Without the contract flag, fmul and fsub stay separate.
define void @boo(ptr %agg.result, ptr %acc, ptr %a, ptr %b) {
entry:
  %0 = load <4 x float>, ptr %a, align 16
  %1 = load <4 x float>, ptr %b, align 16
  %2 = call <4 x float> @llvm.mips.fmul.w(<4 x float> %0, <4 x float> %1)
  %3 = load <4 x float>, ptr %acc, align 16
  %4 = call <4 x float> @llvm.mips.fsub.w(<4 x float> %3, <4 x float> %2)
  store <4 x float> %4, ptr %agg.result, align 16
  ret void
  ; CHECK-LABEL: boo:
  ; CHECK: fmul.w
  ; CHECK: fsub.w
}

; With the contract flag, fmul and fadd fuse into fmadd.
define void @foo_contract(ptr %agg.result, ptr %acc, ptr %a, ptr %b) {
entry:
  %0 = load <4 x float>, ptr %a, align 16
  %1 = load <4 x float>, ptr %b, align 16
  %2 = call contract <4 x float> @llvm.mips.fmul.w(<4 x float> %0, <4 x float> %1)
  %3 = load <4 x float>, ptr %acc, align 16
  %4 = call contract <4 x float> @llvm.mips.fadd.w(<4 x float> %3, <4 x float> %2)
  store <4 x float> %4, ptr %agg.result, align 16
  ret void
  ; CHECK-LABEL: foo_contract:
  ; CHECK: fmadd.w
}

; With the contract flag, fmul and fsub fuse into fmsub.
define void @boo_contract(ptr %agg.result, ptr %acc, ptr %a, ptr %b) {
entry:
  %0 = load <4 x float>, ptr %a, align 16
  %1 = load <4 x float>, ptr %b, align 16
  %2 = call contract <4 x float> @llvm.mips.fmul.w(<4 x float> %0, <4 x float> %1)
  %3 = load <4 x float>, ptr %acc, align 16
  %4 = call contract <4 x float> @llvm.mips.fsub.w(<4 x float> %3, <4 x float> %2)
  store <4 x float> %4, ptr %agg.result, align 16
  ret void
  ; CHECK-LABEL: boo_contract:
  ; CHECK: fmsub.w
}
