; RUN: llc -mtriple=x86_64-apple-darwin10 -O0 < %s | FileCheck %s

; test that we print a label that we use. We had a bug where
; we would print the jump, but not the label because it was considered
; a fall through.

; CHECK:        jmp     LBB0_9
; CHECK: LBB0_9:                                 ## %cleanup

define void @foo()  {
entry:
  br i1 undef, label %land.lhs.true, label %if.end11

land.lhs.true:                                    ; preds = %entry
  br i1 undef, label %if.then, label %if.end11

if.then:                                          ; preds = %land.lhs.true
  br i1 undef, label %if.then9, label %if.end

if.then9:                                         ; preds = %if.then
  br label %cleanup

if.end:                                           ; preds = %if.then
  br label %cleanup

cleanup:                                          ; preds = %if.end, %if.then9
  switch i32 undef, label %default [
    i32 0, label %cleanup.cont
    i32 1, label %if.end11
  ]

cleanup.cont:                                     ; preds = %cleanup
  br label %if.end11

if.end11:                                         ; preds = %cleanup.cont, %cleanup, %land.lhs.true, %entry
  ret void

default:                                          ; preds = %cleanup
  br label %if.end11
}
