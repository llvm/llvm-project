; RUN: mlir-translate -import-llvm %s | FileCheck %s

; A strictfp call site of a named intrinsic becomes `llvm.strictfp`.

; CHECK-LABEL: llvm.func @strictfp_intrinsics(
define float @strictfp_intrinsics(float %x, ptr %dst, ptr %src, i64 %len) strictfp {
  ; CHECK: llvm.intr.fabs(%{{.*}}) {llvm.strictfp} : (f32) -> f32
  %1 = call float @llvm.fabs.f32(float %x) strictfp
  ; CHECK: "llvm.intr.memcpy"({{.*}}) <{isVolatile = false}> {llvm.strictfp}
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %len, i1 false) strictfp
  ; CHECK: llvm.intr.trap {llvm.strictfp}
  call void @llvm.trap() strictfp
  ; CHECK: llvm.intr.fabs(%{{.*}}) : (f32) -> f32
  %2 = call float @llvm.fabs.f32(float %1)
  ret float %2
}
