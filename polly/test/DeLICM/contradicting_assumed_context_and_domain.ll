; RUN: opt %loadPolly -polly-print-delicm -disable-output < %s | FileCheck %s
;
; The domain of bb14 contradicts the SCoP's assumptions. This leads to
; 'anything goes' inside the statement since it is never executed,
; including changing a memory write inside to
;   [p_0, arg1] -> { Stmt_bb14[i0] -> MemRef_tmp[o0] : false }
; (i.e.: never write)
;
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define void @f(ptr %arg, i32 %arg1) {
bb:
  %tmp = alloca [24 x i32], align 4
  br label %bb2

bb2:
  %tmp3 = phi i32 [ 0, %bb ], [ %tmp32, %bb34 ]
  br i1 true, label %bb5, label %bb4

bb4:
  br label %bb24

bb5:
  %tmp6 = sub nsw i32 %arg1, %tmp3
  %tmp7 = add i32 %tmp6, -1
  %tmp8 = icmp eq i32 %tmp3, 0
  br i1 %tmp8, label %bb13, label %bb10

bb10:
  %tmp11 = getelementptr inbounds i16, ptr %arg, i32 %tmp7
  %tmp12 = load i16, ptr %tmp11, align 2
  br label %bb14

bb13:
  br label %bb31

bb14:
  %tmp15 = phi i32 [ 0, %bb10 ], [ %tmp21, %bb14 ]
  %tmp16 = phi i16 [ undef, %bb10 ], [ %tmp19, %bb14 ]
  %tmp17 = getelementptr inbounds [24 x i32], ptr %tmp, i32 0, i32 %tmp15
  %tmp19 = load i16, ptr %arg, align 2
  store i32 undef, ptr %tmp17, align 4
  %tmp20 = call i32 asm "#", "=r,r"(i16 %tmp19) readnone
  %tmp21 = add nuw nsw i32 %tmp15, 1
  %tmp22 = icmp eq i32 %tmp21, %tmp3
  br i1 %tmp22, label %bb23, label %bb14

bb23:
  br label %bb31

bb24:
  %tmp25 = phi i32 [ %tmp30, %bb24 ], [ 0, %bb4 ]
  %tmp26 = mul nsw i32 %tmp25, %arg1
  %tmp27 = getelementptr inbounds i16, ptr %arg, i32 %tmp26
  %tmp29 = load i16, ptr %tmp27, align 2
  %tmp30 = add nuw nsw i32 %tmp25, 1
  br i1 false, label %bb31, label %bb24

bb31:
  %tmp32 = add nuw nsw i32 %tmp3, 1
  br i1 undef, label %bb34, label %bb33

bb33:
  unreachable

bb34:
  br label %bb2
}


; CHECK:      Stmt_bb14
; CHECK:        MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:            [p_0, arg1] -> { Stmt_bb14[i0] -> MemRef_tmp16__phi[] };
; CHECK-NEXT:       new: [p_0, arg1] -> { Stmt_bb14[i0] -> MemRef_tmp[o0] : false };
