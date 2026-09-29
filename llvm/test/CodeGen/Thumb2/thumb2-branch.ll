; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+thumb2 -arm-atomic-cfg-tidy=0 | FileCheck %s
; If-conversion defeats the purpose of this test, which is to check
; conditional branch generation, so a call to make sure it doesn't
; happen and we get actual branches.

declare void @foo()

define i32 @f1(i32 %a, i32 %b, ptr %v) {
entry:
; CHECK-LABEL: f1:
; CHECK: bne LBB
        %tmp = icmp eq i32 %a, %b               ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        call void @foo()
        store i32 0, ptr %v
        ret i32 0

return:         ; preds = %entry
        call void @foo()
        ret i32 1
}

define i32 @f2(i32 %a, i32 %b, ptr %v) {
entry:
; CHECK-LABEL: f2:
; CHECK: bge LBB
        %tmp = icmp slt i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        call void @foo()
        store i32 0, ptr %v
        ret i32 0

return:         ; preds = %entry
        call void @foo()
        ret i32 1
}

define i32 @f3(i32 %a, i32 %b, ptr %v) {
entry:
; CHECK-LABEL: f3:
; CHECK: bhs LBB
        %tmp = icmp ult i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        call void @foo()
        store i32 0, ptr %v
        ret i32 0

return:         ; preds = %entry
        call void @foo()
        ret i32 1
}

define i32 @f4(i32 %a, i32 %b, ptr %v) {
entry:
; CHECK-LABEL: f4:
; CHECK: blo LBB
        %tmp = icmp uge i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        call void @foo()
        store i32 0, ptr %v
        ret i32 0

return:         ; preds = %entry
        call void @foo()
        ret i32 1
}
