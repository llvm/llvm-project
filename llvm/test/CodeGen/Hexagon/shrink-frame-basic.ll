; RUN: llc < %s | FileCheck %s
; Check for allocframe in a non-entry block LBB0_n.
; CHECK: LBB0_{{[0-9]+}}:
; CHECK:   allocframe
; Deallocframe may be in a different block, but must follow.
; CHECK: deallocframe

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define i32 @foo(i32 %n, ptr %p) #0 {
entry:
  %cmp = icmp eq ptr %p, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32, ptr %p, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %p, align 4
  br label %return

if.end:                                           ; preds = %entry
  %call = tail call i32 @bar(i32 %n) #0
  %add = add nsw i32 %call, 1
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ %0, %if.then ], [ %add, %if.end ]
  ret i32 %retval.0
}

declare i32 @bar(...) #0

attributes #0 = { nounwind }

