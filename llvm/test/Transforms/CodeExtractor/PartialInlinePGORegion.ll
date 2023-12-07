; RUN: opt -S -passes=partial-inliner -min-block-execution=1 -skip-partial-inlining-cost-analysis < %s | FileCheck %s
; Require a dummy block (if.then.b) as successor to if.then due to PI requirement
; of region containing more than one BB.
define signext i32 @bar(i32 signext %value, i32 signext %ub) #0 !prof !30 {
entry:
  %value.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  %sum = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %value, ptr %value.addr, align 4
  store i32 %ub, ptr %ub.addr, align 4
  store i32 0, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %ub.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end, !prof !31

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %value.addr, align 4
  %rem = srem i32 %2, 20
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %if.else, !prof !32

if.then:                                          ; preds = %for.body
  %3 = load i32, ptr %value.addr, align 4
  %4 = load i32, ptr %i, align 4
  %mul = mul nsw i32 %4, 5
  %add = add nsw i32 %3, %mul
  %5 = load i32, ptr %sum, align 4
  %add2 = add nsw i32 %5, %add
  store i32 %add2, ptr %sum, align 4
  br label %if.then.b

if.then.b:                                        ; preds = %if.then
  br label %if.end

if.else:                                          ; preds = %for.body
  %6 = load i32, ptr %value.addr, align 4
  %7 = load i32, ptr %i, align 4
  %sub = sub nsw i32 %6, %7
  %8 = load i32, ptr %sum, align 4
  %add3 = add nsw i32 %8, %sub
  store i32 %add3, ptr %sum, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %9 = load i32, ptr %i, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %10 = load i32, ptr %sum, align 4
  ret i32 %10
}

define signext i32 @foo(i32 signext %value, i32 signext %ub) #0 !prof !30 {
; CHECK-LABEL: @foo
; CHECK: codeRepl.i:
; CHECK-NOT: call signext i32 @bar
; CHECK: call void @bar.1.if.then
entry:
  %value.addr = alloca i32, align 4
  %ub.addr = alloca i32, align 4
  store i32 %value, ptr %value.addr, align 4
  store i32 %ub, ptr %ub.addr, align 4
  %0 = load i32, ptr %value.addr, align 4
  %1 = load i32, ptr %ub.addr, align 4
  %call = call signext i32 @bar(i32 signext %0, i32 signext %1)
  ret i32 %call
}

; CHECK-LABEL: define internal void @bar.1.if.then
; CHECK: .exitStub:
; CHECK: ret void

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!29}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 1, !"ProfileSummary", !3}
!3 = !{!4, !5, !6, !7, !8, !9, !10, !11}
!4 = !{!"ProfileFormat", !"InstrProf"}
!5 = !{!"TotalCount", i64 103}
!6 = !{!"MaxCount", i64 100}
!7 = !{!"MaxInternalCount", i64 1}
!8 = !{!"MaxFunctionCount", i64 100}
!9 = !{!"NumCounts", i64 5}
!10 = !{!"NumFunctions", i64 3}
!11 = !{!"DetailedSummary", !12}
!12 = !{!13, !14, !15, !16, !17, !18, !18, !19, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28}
!13 = !{i32 10000, i64 100, i32 1}
!14 = !{i32 100000, i64 100, i32 1}
!15 = !{i32 200000, i64 100, i32 1}
!16 = !{i32 300000, i64 100, i32 1}
!17 = !{i32 400000, i64 100, i32 1}
!18 = !{i32 500000, i64 100, i32 1}
!19 = !{i32 600000, i64 100, i32 1}
!20 = !{i32 700000, i64 100, i32 1}
!21 = !{i32 800000, i64 100, i32 1}
!22 = !{i32 900000, i64 100, i32 1}
!23 = !{i32 950000, i64 100, i32 1}
!24 = !{i32 990000, i64 1, i32 4}
!25 = !{i32 999000, i64 1, i32 4}
!26 = !{i32 999900, i64 1, i32 4}
!27 = !{i32 999990, i64 1, i32 4}
!28 = !{i32 999999, i64 1, i32 4}
!29 = !{!"clang version 6.0.0 (123456)"}
!30 = !{!"function_entry_count", i64 2}
!31 = !{!"branch_weights", i32 100, i32 1}
!32 = !{!"branch_weights", i32 0, i32 100}
