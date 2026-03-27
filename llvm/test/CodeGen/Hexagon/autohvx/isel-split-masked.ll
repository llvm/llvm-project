; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this compiles successfully.
; CHECK: vmem

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @f0() #0 {
b0:
  %v0 = call <64 x i32> @llvm.masked.load.v64i32.p0(ptr nonnull undef, i32 4, <64 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, <64 x i32> undef)
  %v1 = icmp sgt <64 x i32> %v0, zeroinitializer
  %v2 = sext <64 x i1> %v1 to <64 x i32>
  %v3 = add nsw <64 x i32> zeroinitializer, %v2
  %v4 = add nsw <64 x i32> %v3, zeroinitializer
  %v5 = icmp sgt <64 x i32> %v4, zeroinitializer
  %v6 = select <64 x i1> %v5, <64 x i32> %v4, <64 x i32> zeroinitializer
  %v7 = select <64 x i1> zeroinitializer, <64 x i32> undef, <64 x i32> %v6
  %v8 = trunc <64 x i32> %v7 to <64 x i16>
  call void @llvm.masked.store.v64i16.p0(<64 x i16> %v8, ptr undef, i32 2, <64 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>)
  ret void
}

; Function Attrs: argmemonly nounwind readonly willreturn
declare <64 x i32> @llvm.masked.load.v64i32.p0(ptr, i32 immarg, <64 x i1>, <64 x i32>) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.masked.store.v64i16.p0(<64 x i16>, ptr, i32 immarg, <64 x i1>) #2

attributes #0 = { "target-features"="+hvx-length128b,+hvxv67,+v67,-long-calls" }
attributes #1 = { argmemonly nounwind readonly willreturn }
attributes #2 = { argmemonly nounwind willreturn }
