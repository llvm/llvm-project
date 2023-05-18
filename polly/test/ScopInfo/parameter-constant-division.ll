; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops \
; RUN:     -polly-invariant-load-hoisting=true \
; RUN:     -disable-output < %s | FileCheck %s
;
; CHECK:          Invariant Accesses: {
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_land_lhs_true563[] -> MemRef_tmp0[809] };
; CHECK-NEXT:            Execution Context: {  :  }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_if_then570[] -> MemRef_fs[5] };
; CHECK-NEXT:            Execution Context: {  :  }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_if_then570[] -> MemRef_fs[7] };
; CHECK-NEXT:            Execution Context: {  :  }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_if_then570[] -> MemRef_tmp8[813] };
; CHECK-NEXT:            Execution Context: {  :  }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_if_then570[] -> MemRef_tmp3[813] };
; CHECK-NEXT:            Execution Context: {  :  }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_if_then570[] -> MemRef_tmp5[813] };
; CHECK-NEXT:            Execution Context: {  :  }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_if_then570[] -> MemRef_tmp3[812] };
; CHECK-NEXT:            Execution Context: {  :  }
; CHECK-NEXT:    }
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.frame_store = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr }
%struct.picture = type { i32, i32, i32, i32, i32, i32, [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32 }

; Function Attrs: nounwind uwtable
define void @dpb_split_field(ptr %fs) #0 {
entry:
  %frame = getelementptr inbounds %struct.frame_store, ptr %fs, i64 0, i32 10
  br label %for.cond538.preheader.lr.ph

for.cond538.preheader.lr.ph:                      ; preds = %entry
  %bottom_field578 = getelementptr inbounds %struct.frame_store, ptr %fs, i64 0, i32 12
  br label %for.cond538.preheader

for.cond538.preheader:                            ; preds = %for.inc912, %for.cond538.preheader.lr.ph
  %tmp0 = phi ptr [ undef, %for.cond538.preheader.lr.ph ], [ %tmp11, %for.inc912 ]
  br i1 undef, label %land.lhs.true563, label %for.inc912

land.lhs.true563:                                 ; preds = %for.cond538.preheader
  %div552 = sdiv i32 0, 16
  %div554 = sdiv i32 0, 4
  %mul555 = mul i32 %div552, %div554
  %rem558 = srem i32 0, 2
  %tmp9a = add i32 %mul555, 0
  %tmp10a = shl i32 %tmp9a, 1
  %add559 = add i32 %tmp10a, %rem558
  %idxprom564 = sext i32 %add559 to i64
  %mb_field566 = getelementptr inbounds %struct.picture, ptr %tmp0, i64 0, i32 31
  %tmp1 = load ptr, ptr %mb_field566, align 8
  %arrayidx567 = getelementptr inbounds i8, ptr %tmp1, i64 %idxprom564
  %tmp2 = load i8, ptr %arrayidx567, align 1
  store i8 0, ptr %arrayidx567
  br i1 false, label %if.end908, label %if.then570

if.then570:                                       ; preds = %land.lhs.true563
  %tmp3 = load ptr, ptr %frame, align 8
  %mv = getelementptr inbounds %struct.picture, ptr %tmp3, i64 0, i32 35
  %tmp4 = load ptr, ptr %mv, align 8
  %tmp5 = load ptr, ptr %bottom_field578, align 8
  %mv612 = getelementptr inbounds %struct.picture, ptr %tmp5, i64 0, i32 35
  %tmp6 = load ptr, ptr %mv612, align 8
  %arrayidx647 = getelementptr inbounds ptr, ptr %tmp4, i64 1
  %ref_id726 = getelementptr inbounds %struct.picture, ptr %tmp3, i64 0, i32 34
  %tmp7 = load ptr, ptr %ref_id726, align 8
  %arrayidx746 = getelementptr inbounds ptr, ptr %tmp7, i64 5
  %tmp8 = load ptr, ptr %frame, align 8
  %mv783 = getelementptr inbounds %struct.picture, ptr %tmp8, i64 0, i32 35
  %tmp9 = load ptr, ptr %mv783, align 8
  %arrayidx804 = getelementptr inbounds ptr, ptr %tmp9, i64 1
  %tmp10 = load ptr, ptr %arrayidx804, align 8
  store ptr %tmp10, ptr %arrayidx804
  br label %if.end908

if.end908:                                        ; preds = %if.then570, %land.lhs.true563
  br label %for.inc912

for.inc912:                                       ; preds = %if.end908, %for.cond538.preheader
  %tmp11 = phi ptr [ %tmp0, %for.cond538.preheader ], [ undef, %if.end908 ]
  br i1 undef, label %for.cond538.preheader, label %for.cond1392.preheader

for.cond1392.preheader:                           ; preds = %for.inc912
  ret void
}
