; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define void @t() nounwind ssp {
; CHECK: t:
entry:
  br i1 undef, label %return, label %if.end.i

if.end.i:                                         ; preds = %entry
  %tmp7.i = load i32* undef, align 4, !tbaa !0
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %if.end.i
; CHECK: %if.end
; CHECK: movl (%{{.*}}), [[REG:%[a-z]+]]
; CHECK-NOT: movl [[REG]], [[REG]]
; CHECK-NEXT: testl [[REG]], [[REG]]
; CHECK-NEXT: xorb
  %tmp138 = select i1 undef, i32 0, i32 %tmp7.i
  %tmp867 = zext i32 %tmp138 to i64
  br label %while.cond

while.cond:                                       ; preds = %while.body, %if.end
  %tmp869 = sub i64 %tmp867, 0
  %scale2.0 = trunc i64 %tmp869 to i32
  %cmp149 = icmp eq i32 %scale2.0, 0
  br i1 %cmp149, label %while.end, label %land.rhs

land.rhs:                                         ; preds = %while.cond
  br i1 undef, label %while.body, label %while.end

while.body:                                       ; preds = %land.rhs
  br label %while.cond

while.end:                                        ; preds = %land.rhs, %while.cond
  br i1 undef, label %cond.false205, label %cond.true190

cond.true190:                                     ; preds = %while.end
  br i1 undef, label %cond.false242, label %cond.true225

cond.false205:                                    ; preds = %while.end
  unreachable

cond.true225:                                     ; preds = %cond.true190
  br i1 undef, label %cond.false280, label %cond.true271

cond.false242:                                    ; preds = %cond.true190
  unreachable

cond.true271:                                     ; preds = %cond.true225
  unreachable

cond.false280:                                    ; preds = %cond.true225
  unreachable

return:                                           ; preds = %if.end.i, %entry
  ret void
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
