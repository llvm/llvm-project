; RUN: llc -mtriple=hexagon < %s | FileCheck %s
;
; Several shufflevector instructions have masks that are shorter than the
; source vectors. They "gather" a subset of the input elements into a single
; vector. Make sure that they are not expanded into a sequence of extract/
; insert operations.
;
; The C source:
;
; void fred(int *a, int *b, int n) {
;   for (int i = 0; i != n; i += 2) {
;     a[i] += b[i+1];
;     a[i+1] += b[i];
;   }
; }
;
; Command line:
; clang -target hexagon -mcpu=hexagonv60 -fvectorize -fno-unroll-loops -O2 \
;       -mhvx -mhvx-length=128b -S inp.c
;
; CHECK-NOT: vinsert

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: norecurse nounwind
define void @f0(ptr nocapture %a0, ptr nocapture readonly %a1, i32 %a2) #0 {
b0:
  %v0 = icmp eq i32 %a2, 0
  br i1 %v0, label %b7, label %b1

b1:                                               ; preds = %b0
  %v1 = add i32 %a2, -2
  %v2 = lshr i32 %v1, 1
  %v3 = add nuw i32 %v2, 1
  %v4 = icmp ult i32 %v3, 32
  br i1 %v4, label %b2, label %b3

b2:                                               ; preds = %b6, %b3, %b1
  %v5 = phi i32 [ 0, %b3 ], [ 0, %b1 ], [ %v13, %b6 ]
  br label %b8

b3:                                               ; preds = %b1
  %v6 = and i32 %a2, -2
  %v7 = getelementptr i32, ptr %a0, i32 %v6
  %v8 = getelementptr i32, ptr %a1, i32 %v6
  %v9 = icmp ugt ptr %v8, %a0
  %v10 = icmp ugt ptr %v7, %a1
  %v11 = and i1 %v9, %v10
  br i1 %v11, label %b2, label %b4

b4:                                               ; preds = %b3
  %v12 = and i32 %v3, -32
  %v13 = shl i32 %v12, 1
  br label %b5

b5:                                               ; preds = %b5, %b4
  %v14 = phi i32 [ 0, %b4 ], [ %v34, %b5 ]
  %v15 = shl i32 %v14, 1
  %v16 = or i32 %v15, 1
  %v17 = getelementptr inbounds i32, ptr %a1, i32 -1
  %v18 = getelementptr inbounds i32, ptr %v17, i32 %v16
  %v20 = load <64 x i32>, ptr %v18, align 4, !tbaa !1
  %v21 = shufflevector <64 x i32> %v20, <64 x i32> undef, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62>
  %v22 = shufflevector <64 x i32> %v20, <64 x i32> undef, <32 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63>
  %v23 = getelementptr inbounds i32, ptr %a0, i32 %v15
  %v25 = load <64 x i32>, ptr %v23, align 4, !tbaa !1
  %v26 = shufflevector <64 x i32> %v25, <64 x i32> undef, <32 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30, i32 32, i32 34, i32 36, i32 38, i32 40, i32 42, i32 44, i32 46, i32 48, i32 50, i32 52, i32 54, i32 56, i32 58, i32 60, i32 62>
  %v27 = shufflevector <64 x i32> %v25, <64 x i32> undef, <32 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 33, i32 35, i32 37, i32 39, i32 41, i32 43, i32 45, i32 47, i32 49, i32 51, i32 53, i32 55, i32 57, i32 59, i32 61, i32 63>
  %v28 = add nsw <32 x i32> %v26, %v22
  %v29 = getelementptr inbounds i32, ptr %a0, i32 -1
  %v30 = add nsw <32 x i32> %v27, %v21
  %v31 = getelementptr inbounds i32, ptr %v29, i32 %v16
  %v33 = shufflevector <32 x i32> %v28, <32 x i32> %v30, <64 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  store <64 x i32> %v33, ptr %v31, align 4, !tbaa !1
  %v34 = add i32 %v14, 32
  %v35 = icmp eq i32 %v34, %v12
  br i1 %v35, label %b6, label %b5, !llvm.loop !5

b6:                                               ; preds = %b5
  %v36 = icmp eq i32 %v3, %v12
  br i1 %v36, label %b7, label %b2

b7:                                               ; preds = %b8, %b6, %b0
  ret void

b8:                                               ; preds = %b8, %b2
  %v37 = phi i32 [ %v49, %b8 ], [ %v5, %b2 ]
  %v38 = or i32 %v37, 1
  %v39 = getelementptr inbounds i32, ptr %a1, i32 %v38
  %v40 = load i32, ptr %v39, align 4, !tbaa !1
  %v41 = getelementptr inbounds i32, ptr %a0, i32 %v37
  %v42 = load i32, ptr %v41, align 4, !tbaa !1
  %v43 = add nsw i32 %v42, %v40
  store i32 %v43, ptr %v41, align 4, !tbaa !1
  %v44 = getelementptr inbounds i32, ptr %a1, i32 %v37
  %v45 = load i32, ptr %v44, align 4, !tbaa !1
  %v46 = getelementptr inbounds i32, ptr %a0, i32 %v38
  %v47 = load i32, ptr %v46, align 4, !tbaa !1
  %v48 = add nsw i32 %v47, %v45
  store i32 %v48, ptr %v46, align 4, !tbaa !1
  %v49 = add nuw nsw i32 %v37, 2
  %v50 = icmp eq i32 %v49, %a2
  br i1 %v50, label %b7, label %b8, !llvm.loop !7
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length128b,+hvxv60" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.isvectorized", i32 1}
!7 = distinct !{!7, !6}
