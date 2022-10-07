; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s

declare void @cxa_atexit_check_1(ptr)

; TODO: KB: ORiginal test case was just checking it compiles; is this worth keeping?
; CHECK: check_cxa_atexit:
; CHECK: blr

define i32 @check_cxa_atexit(ptr %cxa_atexit, ptr %cxa_finalize) {
entry:
        %tmp7 = call i32 null( ptr @cxa_atexit_check_1, ptr null, ptr null )            ; <i32> [#uses=0]
        br i1 false, label %cond_true, label %cond_next

cond_true:    ; preds = %entry
        ret i32 0

cond_next:        ; preds = %entry
        ret i32 0
}
