; RUN: opt -S -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 < %s | FileCheck %s

@global = local_unnamed_addr global i32 0, align 4
@global.1 = local_unnamed_addr global i32 0, align 4
@global.2 = local_unnamed_addr global float 0x3EF0000000000000, align 4

; CHECK-LABEL: @PR33706
; CHECK-NOT: <2 x i32>
define void @PR33706(float* nocapture readonly %arg, float* nocapture %arg1, i32 %arg2) local_unnamed_addr {
bb:
  %tmp = load i32, i32* @global.1, align 4
  %tmp3 = getelementptr inbounds float, float* %arg, i64 190
  %tmp4 = getelementptr inbounds float, float* %arg1, i64 512
  %tmp5 = and i32 %tmp, 65535
  %tmp6 = icmp ugt i32 %arg2, 65536
  br i1 %tmp6, label %bb7, label %bb9

bb7:                                              ; preds = %bb
  %tmp8 = load i32, i32* @global, align 4
  br label %bb27

bb9:                                              ; preds = %bb
  %tmp10 = udiv i32 65536, %arg2
  br label %bb11

bb11:                                             ; preds = %bb11, %bb9
  %tmp12 = phi i32 [ %tmp20, %bb11 ], [ %tmp5, %bb9 ]
  %tmp13 = phi float* [ %tmp18, %bb11 ], [ %tmp4, %bb9 ]
  %tmp14 = phi i32 [ %tmp16, %bb11 ], [ %tmp10, %bb9 ]
  %tmp15 = phi i32 [ %tmp19, %bb11 ], [ %tmp, %bb9 ]
  %tmp16 = add nsw i32 %tmp14, -1
  %tmp17 = sitofp i32 %tmp12 to float
  store float %tmp17, float* %tmp13, align 4
  %tmp18 = getelementptr inbounds float, float* %tmp13, i64 1
  %tmp19 = add i32 %tmp15, %arg2
  %tmp20 = and i32 %tmp19, 65535
  %tmp21 = icmp eq i32 %tmp16, 0
  br i1 %tmp21, label %bb22, label %bb11

bb22:                                             ; preds = %bb11
  %tmp23 = phi float* [ %tmp18, %bb11 ]
  %tmp24 = phi i32 [ %tmp19, %bb11 ]
  %tmp25 = phi i32 [ %tmp20, %bb11 ]
  %tmp26 = ashr i32 %tmp24, 16
  store i32 %tmp26, i32* @global, align 4
  br label %bb27

bb27:                                             ; preds = %bb22, %bb7
  %tmp28 = phi i32 [ %tmp26, %bb22 ], [ %tmp8, %bb7 ]
  %tmp29 = phi float* [ %tmp23, %bb22 ], [ %tmp4, %bb7 ]
  %tmp30 = phi i32 [ %tmp25, %bb22 ], [ %tmp5, %bb7 ]
  %tmp31 = sext i32 %tmp28 to i64
  %tmp32 = getelementptr inbounds float, float* %tmp3, i64 %tmp31
  %tmp33 = load float, float* %tmp32, align 4
  %tmp34 = sitofp i32 %tmp30 to float
  %tmp35 = load float, float* @global.2, align 4
  %tmp36 = fmul float %tmp35, %tmp34
  %tmp37 = fadd float %tmp33, %tmp36
  store float %tmp37, float* %tmp29, align 4
  ret void
}
