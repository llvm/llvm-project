; RUN: llc -mtriple=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -hexagon-hvx-widen=16 < %s | FileCheck %s

; Check for successful compilation.
; CHECK-LABEL: f0:
; CHECK: vmemu

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local void @f0(ptr %a0) local_unnamed_addr #0 {
b0:
  %v0 = getelementptr i16, ptr %a0, i32 8
  %v2 = icmp eq i32 0, 0
  %v3 = insertelement <8 x i1> undef, i1 %v2, i64 0
  %v4 = shufflevector <8 x i1> %v3, <8 x i1> undef, <8 x i32> zeroinitializer
  %v5 = call <8 x i32> @llvm.masked.load.v8i32.p0(ptr nonnull undef, i32 4, <8 x i1> %v4, <8 x i32> undef)
  %v6 = sub nsw <8 x i32> zeroinitializer, %v5
  %v7 = add nsw <8 x i32> %v6, zeroinitializer
  %v8 = add <8 x i32> zeroinitializer, %v7
  %v9 = lshr <8 x i32> %v8, <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  %v10 = trunc <8 x i32> %v9 to <8 x i16>
  call void @llvm.masked.store.v8i16.p0(<8 x i16> %v10, ptr %v0, i32 2, <8 x i1> %v4)
  ret void
}

; Function Attrs: argmemonly nounwind readonly willreturn
declare <8 x i32> @llvm.masked.load.v8i32.p0(ptr, i32 immarg, <8 x i1>, <8 x i32>) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.masked.store.v8i16.p0(<8 x i16>, ptr, i32 immarg, <8 x i1>) #2

attributes #0 = { "target-features"="+hvx-length64b,+hvxv65,+v65,-long-calls,-packets" }
attributes #1 = { argmemonly nounwind readonly willreturn }
attributes #2 = { argmemonly nounwind willreturn }
