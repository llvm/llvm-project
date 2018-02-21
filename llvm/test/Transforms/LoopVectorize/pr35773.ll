; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -scev-version-unknown < %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
@a = common local_unnamed_addr global i32 0, align 4
@b = common local_unnamed_addr global i8 0, align 1

; Function Attrs: norecurse nounwind uwtable
define void @doit1() local_unnamed_addr{
entry:
  br label %for.body

for.body:
  %main.iv = phi i32 [ 0, %entry ], [ %inc, %for.body ]

  %i8.iv = phi i8 [ 0, %entry ], [ %i8.add, %for.body ]
  %i32.iv = phi i32 [ 0, %entry ], [ %i32.add, %for.body ]

  %trunc.to.be.converted.to.new.iv = trunc i32 %i32.iv to i8
  %i8.add = add i8 %i8.iv, %trunc.to.be.converted.to.new.iv

  %noop.conv.under.pse = and i32 %i32.iv, 255
  %i32.add = add nuw nsw i32 %noop.conv.under.pse, 9

  %inc = add i32 %main.iv, 1
  %tobool = icmp eq i32 %inc, 16
  br i1 %tobool, label %for.cond.for.end_crit_edge, label %for.body

; CHECK-LABEL: @doit1(
; CHECK:       vector.body:
; CHECK-NEXT:    [[MAIN_IV:%.*]] = phi i32 [ 0, [[VECTOR_PH:%.*]] ], [ [[MAIN_IV_NEXT:%.*]], [[VECTOR_BODY:%.*]] ]
; CHECK-NEXT:    [[I8_IV:%.*]] = phi <4 x i8> [ zeroinitializer, [[VECTOR_PH]] ], [ [[I8_IV_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[I32_IV:%.*]] = phi <4 x i32> [ <i32 0, i32 9, i32 18, i32 27>, [[VECTOR_PH]] ], [ [[I32_IV_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[IV_FROM_TRUNC:%.*]] = phi <4 x i8> [ <i8 0, i8 9, i8 18, i8 27>, [[VECTOR_PH]] ], [ [[IV_FROM_TRUNC_NEXT:%.*]], [[VECTOR_BODY]] ]

; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> undef, i32 [[MAIN_IV]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <4 x i32> [[BROADCAST_SPLAT]], <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    [[TMP7:%.*]] = add i32 [[MAIN_IV]], 0

; CHECK-NEXT:    [[I8_IV_NEXT]] = add <4 x i8> [[I8_IV]], [[IV_FROM_TRUNC]]

; CHECK-NEXT:    [[MAIN_IV_NEXT]] = add i32 [[MAIN_IV]], 4
; CHECK-NEXT:    [[I32_IV_NEXT]] = add <4 x i32> [[I32_IV]], <i32 36, i32 36, i32 36, i32 36>
; CHECK-NEXT:    [[IV_FROM_TRUNC_NEXT]] = add <4 x i8> [[IV_FROM_TRUNC]], <i8 36, i8 36, i8 36, i8 36>
; CHECK-NEXT:    [[TMP9:%.*]] = icmp eq i32 [[MAIN_IV_NEXT]], 16
; CHECK-NEXT:    br i1 [[TMP9]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop !0

for.cond.for.end_crit_edge:
  store i8 %i8.add, i8* @b, align 1
  br label %for.end

for.end:
  ret void
}
