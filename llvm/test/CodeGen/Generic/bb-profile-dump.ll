; REQUIRES: x86-registered-target
;
; Check that the basic block profile dump outputs data and in the correct
; format.
;
; RUN: llc -mtriple=x86_64-linux-unknown -o /dev/null -basic-block-sections=labels -mbb-profile-dump=- %s | FileCheck %s

; Check that given a simple case, we can return the default MBFI

define i64 @f2(i64 %a, i64 %b) !prof !1{
    %sum = add i64 %a, %b
    ret i64 %sum
}

; CHECK: f2,0,1.000000e+03

define i64 @f1() !prof !2{
    %sum = call i64 @f2(i64 2, i64 2)
    %isEqual = icmp eq i64 %sum, 4
    br i1 %isEqual, label %ifEqual, label %ifNotEqual, !prof !3
ifEqual:
    ret i64 0
ifNotEqual:
    ret i64 %sum
}

; CHECK-NEXT: f1,0,1.000000e+01
; CHECK-NEXT: f1,2,6.000000e+00
; CHECK-NEXT: f1,1,4.000000e+00

define void @f3(i32 %iter) !prof !4 {
entry:
    br label %loop
loop:
    %i = phi i32 [0, %entry], [%i_next, %loop]
    %i_next = add i32 %i, 1
    %exit_cond = icmp slt i32 %i_next, %iter
    br i1 %exit_cond, label %loop, label %exit, !prof !5
exit:
    ret void
}

; CHECK-NEXT: f3,0,2.000000e+00
; CHECK-NEXT: f3,1,2.002000e+03
; CHECK-NEXT: f3,2,2.000000e+00

!1 = !{!"function_entry_count", i64 1000}
!2 = !{!"function_entry_count", i64 10}
!3 = !{!"branch_weights", i32 2, i32 3}
!4 = !{!"function_entry_count", i64 2}
!5 = !{!"branch_weights", i32 1000, i32 1}

; Check that if we pass -mbb-profile-dump but don't set -basic-block-sections,
; we get an appropriate error message

; RUN: not llc -mtriple=x86_64-linux-unknown -o /dev/null -mbb-profile-dump=- %s 2>&1 | FileCheck --check-prefix=NO-SECTIONS %s

; NO-SECTIONS: <unknown>:0: error: Unable to find BB labels for MBB profile dump. -mbb-profile-dump must be called with -basic-block-sections=labels

