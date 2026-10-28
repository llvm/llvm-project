; LLVMToAIRIntrinsics names AIR functions after the intrinsic's type suffix,
; so bfloat math becomes air.<fn>.bf16, and Metal's compiler never finished
; compiling those on an M2 Max (llvm/utils/air-gpu bf16_fabs_bits, bf16_fma;
; PTX-formalization cvt.rni.s8.bf16). The Metal pipeline computes them, and
; bfloat compares, in f32; the portable AIR keeps them.
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE

; CHECK-LABEL: define bfloat @rint_bf16
; CHECK: call float @air.rint.f32(float
; CHECK-NOT: bf16(
; CHECK: ret bfloat
; PORTABLE-LABEL: define bfloat @rint_bf16
; PORTABLE: call bfloat @air.rint.bf16(bfloat
define bfloat @rint_bf16(bfloat %x) {
  %r = call bfloat @llvm.rint.bf16(bfloat %x)
  ret bfloat %r
}

; CHECK-LABEL: define bfloat @fma_bf16
; CHECK: call float @air.fma.f32(float {{%.*}}, float {{%.*}}, float {{%.*}})
define bfloat @fma_bf16(bfloat %a, bfloat %b, bfloat %c) {
  %r = call bfloat @llvm.fma.bf16(bfloat %a, bfloat %b, bfloat %c)
  ret bfloat %r
}

; CHECK-LABEL: define i1 @cmp_bf16
; CHECK: fcmp olt float
; CHECK-NOT: fcmp {{.*}} bfloat
define i1 @cmp_bf16(bfloat %a, bfloat %b) {
  %r = fcmp olt bfloat %a, %b
  ret i1 %r
}

declare bfloat @llvm.rint.bf16(bfloat)
declare bfloat @llvm.fma.bf16(bfloat, bfloat, bfloat)
