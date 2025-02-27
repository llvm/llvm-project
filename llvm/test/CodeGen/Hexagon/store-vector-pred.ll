; RUN: llc -mtriple=hexagon -hexagon-instsimplify=0 -hexagon-masked-vmem=0 < %s | FileCheck %s

; This test checks that store a vector predicate of type v128i1 is lowered
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
  %v3 = insertelement <128 x i32> undef, i32 %v2, i32 0
  %v4 = shufflevector <128 x i32> %v3, <128 x i32> undef, <128 x i32> zeroinitializer
  %v5 = icmp ule <128 x i32> undef, %v4
  %v6 = call <128 x i8> @llvm.masked.load.v128i8.p0(ptr nonnull undef, i32 1, <128 x i1> %v5, <128 x i8> undef)
  %v7 = lshr <128 x i8> %v6, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %v8 = and <128 x i8> %v7, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
  %v9 = zext <128 x i8> %v8 to <128 x i32>
  %v10 = add nsw <128 x i32> undef, %v9
  %v11 = select <128 x i1> %v5, <128 x i32> %v10, <128 x i32> undef
  %v12 = add <128 x i32> %v11, undef
  %v13 = add <128 x i32> %v12, undef
  %v14 = add <128 x i32> %v13, undef
  %v15 = add <128 x i32> %v14, undef
  %v16 = add <128 x i32> %v15, undef
  %v17 = add <128 x i32> %v16, undef
  %v18 = add <128 x i32> %v17, undef
  %v19 = extractelement <128 x i32> %v18, i32 0
  %v20 = getelementptr inbounds i8, ptr null, i32 2160
  store i32 %v19, ptr %v20, align 4
  br label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

; Function Attrs: argmemonly nounwind readonly willreturn
declare <128 x i8> @llvm.masked.load.v128i8.p0(ptr, i32 immarg, <128 x i1>, <128 x i8>) #1

attributes #0 = { "target-features"="+hvx-length128b,+hvxv67,+v67,-long-calls" }
attributes #1 = { argmemonly nounwind readonly willreturn }
