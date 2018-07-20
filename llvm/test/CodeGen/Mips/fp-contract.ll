; Test that the compiled does not fuse fmul and fadd into fmadd when no -fp-contract=fast
; option is set (the same applies for fmul, fsub and fmsub).

; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s --check-prefixes=CHECK-CONTRACT-OFF
; RUN: llc -march=mipsel -mattr=+msa,+fp64 -fp-contract=off < %s | FileCheck %s --check-prefixes=CHECK-CONTRACT-OFF
; RUN: llc -march=mips -mattr=+msa,+fp64 -fp-contract=fast < %s | FileCheck %s --check-prefixes=CHECK-CONTRACT-FAST

declare <4 x float> @llvm.mips.fmul.w(<4 x float>, <4 x float>)
declare <4 x float> @llvm.mips.fadd.w(<4 x float>, <4 x float>)
declare <4 x float> @llvm.mips.fsub.w(<4 x float>, <4 x float>)

define void @foo(<4 x float>* %agg.result, <4 x float>* %acc, <4 x float>* %a, <4 x float>* %b) {
entry:
  %0 = load <4 x float>, <4 x float>* %a, align 16
  %1 = load <4 x float>, <4 x float>* %b, align 16
  %2 = call <4 x float> @llvm.mips.fmul.w(<4 x float> %0, <4 x float> %1)
  %3 = load <4 x float>, <4 x float>* %acc, align 16
  %4 = call <4 x float> @llvm.mips.fadd.w(<4 x float> %3, <4 x float> %2)
  store <4 x float> %4, <4 x float>* %agg.result, align 16
  ret void
  ; CHECK-CONTRACT-OFF: fmul.w
  ; CHECK-CONTRACT-OFF: fadd.w
  ; CHECK-CONTRACT-FAST: fmadd.w
}

define void @boo(<4 x float>* %agg.result, <4 x float>* %acc, <4 x float>* %a, <4 x float>* %b) {
entry:
  %0 = load <4 x float>, <4 x float>* %a, align 16
  %1 = load <4 x float>, <4 x float>* %b, align 16
  %2 = call <4 x float> @llvm.mips.fmul.w(<4 x float> %0, <4 x float> %1)
  %3 = load <4 x float>, <4 x float>* %acc, align 16
  %4 = call <4 x float> @llvm.mips.fsub.w(<4 x float> %3, <4 x float> %2)
  store <4 x float> %4, <4 x float>* %agg.result, align 16
  ret void
  ; CHECK-CONTRACT-OFF: fmul.w
  ; CHECK-CONTRACT-OFF: fsub.w
  ; CHECK-CONTRACT-FAST: fmsub.w
}
