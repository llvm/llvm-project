; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S -inline-threshold=50 -inline-cold-callsite-threshold=0 -hot-callsite-threshold=50 | FileCheck %s
; This tests incremental updates to caller's BFI as a callee gets inlined.
; In bottom-up inlining, first c->e inlining is considered and fails because
; e's size exceeds the threshold of 50. Then a->c inlining is considered and it
; succeeds. a's BFI is updated incrementally. As c's blocks get pruned, the 
; block with label cond_false is removed and since the remanining code is
; straight-line a single block gets cloned into a. This block should get the
; maximum block frequency among the original blocks in c. If it gets the
; frequency of the block with label cond_true in @c, its frequency will be
; 1/10th of function a's entry block frequency, resulting in a callsite count of
; 2 (since a's entry count is 20) which means that a->e callsite will be
; considered cold and not inlined. 

@data = external global i32
; CHECK-LABEL: define i32 @a(
define i32 @a(i32 %a1) !prof !21 {
; CHECK-NOT: call i32 @c
; CHECK-NOT: call i32 @e
; CHECK: ret
entry:
  %cond = icmp sle i32 %a1, 1
  %a2 = call i32 @c(i32 1)
  br label %exit
exit:
  ret i32 %a2
}

declare void @ext();

; CHECK: @c(i32 %c1) !prof [[COUNT1:![0-9]+]]
define i32 @c(i32 %c1) !prof !23 {
  call void @ext()
  %cond = icmp sle i32 %c1, 1
  br i1 %cond, label %cond_true, label %cond_false, !prof !25

cond_false:
  br label %exit

cond_true:
  %c11 = call i32 @e(i32 %c1)
  br label %exit
exit:
  %c12 = phi i32 [ 0, %cond_false], [ %c11, %cond_true ]
  ret i32 %c12
}


; CHECK: @e(i32 %c1) !prof [[COUNT2:![0-9]+]]
define i32 @e(i32 %c1) !prof !24 {
  call void @ext()
  call void @ext()
  %cond = icmp sle i32 %c1, 1
  br i1 %cond, label %cond_true, label %cond_false

cond_false:
  call void @ext()
  %c2 = load i32, ptr @data, align 4
  %c3 = add i32 %c1, %c2
  %c4 = mul i32 %c3, %c2
  %c5 = add i32 %c4, %c2
  %c6 = mul i32 %c5, %c2
  %c7 = add i32 %c6, %c2
  %c8 = mul i32 %c7, %c2
  %c9 = add i32 %c8, %c2
  %c10 = mul i32 %c9, %c2
  ret i32 %c10

cond_true:
  ret i32 0
}

; CHECK: [[COUNT1]] = !{!"function_entry_count", i64 480}
; CHECK: [[COUNT2]] = !{!"function_entry_count", i64 80}
!21 = !{!"function_entry_count", i64 20}
!23 = !{!"function_entry_count", i64 500}
!24 = !{!"function_entry_count", i64 100}
!25 = !{!"branch_weights", i32 1, i32 9}

!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 1000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 1000, i32 1}
!13 = !{i32 999000, i64 1000, i32 1}
!14 = !{i32 999999, i64 5, i32 2}
