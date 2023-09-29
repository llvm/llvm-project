; REQUIRES: x86-registered-target
;
; Check that the basic block profile dump outputs data and in the correct
; format.
;
; RUN: llc -mtriple=x86_64-linux-unknown -o /dev/null -basic-block-sections=labels -mbb-profile-dump=- %s | FileCheck %s

; Check that given a simple case, we can return the default MBFI

define i64 @f2(i64 %a, i64 %b) {
    %sum = add i64 %a, %b
    ret i64 %sum
}

; CHECK: f2,0,8

define i64 @f1() {
    %sum = call i64 @f2(i64 2, i64 2)
    %isEqual = icmp eq i64 %sum, 4
    br i1 %isEqual, label %ifEqual, label %ifNotEqual
ifEqual:
    ret i64 0
ifNotEqual:
    ret i64 %sum
}

; CHECK-NEXT: f1,0,16
; CHECK-NEXT: f1,1,8
; CHECK-NEXT: f1,2,16

; Check that if we pass -mbb-profile-dump but don't set -basic-block-sections,
; we get an appropriate error message

; RUN: not llc -mtriple=x86_64-linux-unknown -o /dev/null -mbb-profile-dump=- %s 2>&1 | FileCheck --check-prefix=NO-SECTIONS %s

; NO-SECTIONS: <unknown>:0: error: Unable to find BB labels for MBB profile dump. -mbb-profile-dump must be called with -basic-block-sections=labels

