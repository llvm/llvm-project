; RUN: opt -simplifycfg -S %s | FileCheck %s
; Make sure we don't speculate loads under ThreadSanitizer.
@g = global i32 0, align 4

define i32 @TestNoTsan(i32 %cond) nounwind readonly uwtable {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %return, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32, i32* @g, align 4
  br label %return

return:                                           ; preds = %entry, %if.then
  %retval = phi i32 [ %0, %if.then ], [ 0, %entry ]
  ret i32 %retval
; CHECK-LABEL: @TestNoTsan
; CHECK: %[[LOAD:[^ ]*]] = load
; CHECK: select{{.*}}[[LOAD]]
; CHECK: ret i32
}

define i32 @TestTsan(i32 %cond) nounwind readonly uwtable sanitize_thread {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %return, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32, i32* @g, align 4
  br label %return

return:                                           ; preds = %entry, %if.then
  %retval = phi i32 [ %0, %if.then ], [ 0, %entry ]
  ret i32 %retval
; CHECK-LABEL: @TestTsan
; CHECK: br i1
; CHECK: load i32, i32* @g
; CHECK: br label
; CHECK: ret i32
}
