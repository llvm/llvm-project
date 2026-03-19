; RUN: llc -mtriple=hexagon < %s | FileCheck %s


; Check for a non-crashing output.
; CHECK: vmem
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @fred() #0 {
  tail call void @llvm.masked.store.v64f16.p0(<64 x half> <half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef, half 0xHFBFF, half undef>, ptr undef, i32 64, <64 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>)
  ret void
}

; Function Attrs: argmemonly nounwind willreturn writeonly
declare void @llvm.masked.store.v64f16.p0(<64 x half>, ptr, i32 immarg, <64 x i1>) #0

attributes #0 = { argmemonly nounwind willreturn writeonly "target-cpu"="hexagonv69" "target-features"="+hvxv69,+hvx-length128b,+hvx-qfloat" }
