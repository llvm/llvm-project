; RUN: opt %loadPolly -polly-scops -analyze \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s --check-prefix=IR
;
; Verify we do not create assumptions based on the parameter p_1 which is the
; load %0 and due to error-assumptions not "part of the SCoP".
;
; CHECK:        Invalid Context:
; CHECK-NEXT:     [releaseCount, p_1] -> {  : releaseCount > 0 }
;
; IR: polly.start
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: uwtable
define void @_ZN8NWindows16NSynchronization14CSemaphoreWFMO7ReleaseEi(i32 %releaseCount) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp = icmp slt i32 %releaseCount, 1
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry.split
  tail call void @_ZN8NWindows16NSynchronization8CSynchro5EnterEv()
  %0 = load i32, i32* null, align 8
  %add = add nsw i32 %0, %releaseCount
  %cmp2 = icmp sgt i32 %add, 0
  br i1 %cmp2, label %if.then3, label %if.end5

if.then3:                                         ; preds = %if.end
  br label %return

if.end5:                                          ; preds = %if.end
  br label %return

return:                                           ; preds = %if.end5, %if.then3, %entry.split
  %retval.1 = phi i32 [ 1, %entry.split ], [ 1, %if.then3 ], [ 0, %if.end5 ]
  ret void
}

; Function Attrs: nounwind uwtable
declare void @_ZN8NWindows16NSynchronization8CSynchro5EnterEv()
