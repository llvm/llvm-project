; RUN: mlir-translate -import-llvm %s | FileCheck %s
; RUN: mlir-translate -import-llvm -mlir-to-llvmir %s | FileCheck %s --check-prefix=ROUNDTRIP

declare void @llvm.assume(i1)
declare float @llvm.fptrunc.round.f32.f64(double, metadata)

; CHECK-LABEL: llvm.func @assume(
; CHECK-SAME: %[[PTR:.*]]: !llvm.ptr)
; CHECK: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
; CHECK: %[[ALIGN:.*]] = llvm.mlir.constant(16 : i64) : i64
; CHECK: llvm.intr.assume %[[TRUE]] ["align"(%[[PTR]], %[[ALIGN]] : !llvm.ptr, i64)] : i1
; ROUNDTRIP-LABEL: define void @assume(
; ROUNDTRIP-SAME: ptr %[[PTR:.*]])
; ROUNDTRIP: call void @llvm.assume(i1 true) [ "align"(ptr %[[PTR]], i64 16) ]
define void @assume(ptr %p) {
  call void @llvm.assume(i1 true) ["align"(ptr %p, i64 16)]
  ret void
}

; CHECK-LABEL: llvm.func @fptrunc_round(
; CHECK-SAME: %[[ARG:.*]]: f64, %[[STATE:.*]]: i32)
; CHECK: %[[ROUNDING:.*]] = llvm.mlir.metadata_as_value #llvm.md_string<"round.tonearest">
; CHECK: %[[RESULT:.*]] = llvm.call_intrinsic "llvm.fptrunc.round.f32.f64"(%[[ARG]], %[[ROUNDING]]) ["tag"(%[[STATE]] : i32), "empty"()] : (f64, !llvm.metadata) -> f32
; CHECK: llvm.return %[[RESULT]] : f32
; ROUNDTRIP-LABEL: define float @fptrunc_round(
; ROUNDTRIP-SAME: double %[[ARG:.*]], i32 %[[STATE:.*]])
; ROUNDTRIP: call float @llvm.fptrunc.round.f32.f64(double %[[ARG]], metadata !"round.tonearest") [ "tag"(i32 %[[STATE]]), "empty"() ]
define float @fptrunc_round(double %arg, i32 %state) {
  %result = call float @llvm.fptrunc.round.f32.f64(double %arg, metadata !"round.tonearest") ["tag"(i32 %state), "empty"()]
  ret float %result
}
