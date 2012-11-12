; RUN: opt < %s -basicaa -globalopt -instcombine -loop-rotate -licm -instcombine -indvars -loop-deletion -constmerge -S
; PR11882: ComputeLoadConstantCompareExitLimit crash.
;
; for.body is deleted leaving a loop-invariant load.
; CHECK-NOT: for.body
target datalayout = "e-p:64:64:64-n32:64"

@func_21_l_773 = external global i32, align 4
@g_814 = external global i32, align 4
@g_244 = internal global [1 x [0 x i32]] zeroinitializer, align 4

define void @func_21() nounwind uwtable ssp {
entry:
  br label %lbl_818

lbl_818:                                          ; preds = %for.end, %entry
  call void (...)* @func_27()
  store i32 0, i32* @g_814, align 4, !tbaa !0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %lbl_818
  %0 = load i32* @g_814, align 4, !tbaa !0
  %cmp = icmp sle i32 %0, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [0 x i32]* getelementptr inbounds ([1 x [0 x i32]]* @g_244, i32 0, i64 0), i32 0, i64 %idxprom
  %1 = load i32* %arrayidx, align 1, !tbaa !0
  store i32 %1, i32* @func_21_l_773, align 4, !tbaa !0
  store i32 1, i32* @g_814, align 4, !tbaa !0
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %2 = load i32* @func_21_l_773, align 4, !tbaa !0
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %lbl_818, label %if.end

if.end:                                           ; preds = %for.end
  ret void
}

declare void @func_27(...)

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
