; RUN: llc -mtriple=hexagon -hexagon-instsimplify=0 -hexagon-masked-vmem=0 < %s | FileCheck %s

; Test that LLVM does not assert and bitcast v64i1 to i64 is lowered
; without crashing.
; CHECK: valign

target triple = "hexagon"

define dso_local void @f0() local_unnamed_addr #0 {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  %v0 = load i8, ptr undef, align 1
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, -1
  %v3 = insertelement <64 x i32> undef, i32 %v2, i32 0
  %v4 = shufflevector <64 x i32> %v3, <64 x i32> undef, <64 x i32> zeroinitializer
  %v5 = icmp ule <64 x i32> undef, %v4
  %v6 = call <64 x i8> @llvm.masked.load.v64i8.p0(ptr nonnull undef, i32 1, <64 x i1> %v5, <64 x i8> undef)
  %v7 = lshr <64 x i8> %v6, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %v8 = and <64 x i8> %v7, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %v9 = zext <64 x i8> %v8 to <64 x i32>
  %v10 = add nsw <64 x i32> undef, %v9
  %v11 = select <64 x i1> %v5, <64 x i32> %v10, <64 x i32> undef
  %v12 = add <64 x i32> %v11, undef
  %v13 = add <64 x i32> %v12, undef
  %v14 = add <64 x i32> %v13, undef
  %v15 = add <64 x i32> %v14, undef
  %v16 = add <64 x i32> %v15, undef
  %v17 = add <64 x i32> %v16, undef
  %v18 = add <64 x i32> %v17, undef
  %v19 = extractelement <64 x i32> %v18, i32 0
  %v20 = getelementptr inbounds i8, ptr null, i32 2160
  store i32 %v19, ptr %v20, align 4
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

; Function Attrs: argmemonly nounwind readonly willreturn
declare <64 x i8> @llvm.masked.load.v64i8.p0(ptr, i32 immarg, <64 x i1>, <64 x i8>) #1

attributes #0 = { "target-features"="+hvx-length64b,+hvxv67,+v67,-long-calls" }
attributes #1 = { argmemonly nounwind readonly willreturn }
