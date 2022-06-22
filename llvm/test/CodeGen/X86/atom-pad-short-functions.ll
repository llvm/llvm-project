; RUN: llc < %s -O1 -mcpu=atom -mtriple=i686-linux  | FileCheck %s

declare void @external_function(...)

define i32 @test_return_val(i32 %a) nounwind {
; CHECK: test_return_val
; CHECK: movl
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: ret
  ret i32 %a
}

define i32 @test_optsize(i32 %a) nounwind optsize {
; CHECK: test_optsize
; CHECK: movl
; CHECK-NEXT: ret
  ret i32 %a
}

define i32 @test_minsize(i32 %a) nounwind minsize {
; CHECK: test_minsize
; CHECK: movl
; CHECK-NEXT: ret
  ret i32 %a
}

define i32 @test_pgso(i32 %a) nounwind !prof !14 {
; CHECK: test_pgso
; CHECK: movl
; CHECK-NEXT: ret
  ret i32 %a
}

define i32 @test_add(i32 %a, i32 %b) nounwind {
; CHECK: test_add
; CHECK: addl
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: ret
  %result = add i32 %a, %b
  ret i32 %result
}

define i32 @test_multiple_ret(i32 %a, i32 %b, i1 %c) nounwind {
; CHECK: @test_multiple_ret
; CHECK: je

; CHECK: nop
; CHECK: nop
; CHECK: ret

; CHECK: nop
; CHECK: nop
; CHECK: ret

  br i1 %c, label %bb1, label %bb2

bb1:
  ret i32 %a

bb2:
  ret i32 %b
}

define void @test_call_others(i32 %x) nounwind
{
; CHECK: test_call_others
; CHECK: je
  %tobool = icmp eq i32 %x, 0
  br i1 %tobool, label %if.end, label %true.case

; CHECK: jmp external_function
true.case:
  tail call void @external_function() nounwind
  br label %if.end

; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: ret
if.end:
  ret void

}

define void @test_branch_to_same_bb(i32 %x, i32 %y) nounwind {
; CHECK: @test_branch_to_same_bb
  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %while.cond, label %while.end

while.cond:
  br label %while.cond

; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: nop
; CHECK: ret
while.end:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
