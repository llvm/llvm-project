; RUN: opt -hotcoldsplit -S < %s | FileCheck %s
; RUN: opt -passes=hotcoldsplit -S < %s | FileCheck %s

; Check that these functions are not split. Outlined functions are called from a
; basic block named codeRepl.

; The cold region is too small to split.
; CHECK-LABEL: @foo
; CHECK-NOT: codeRepl
define void @foo() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  br label %if.then12

if.then12:                                        ; preds = %if.end
  br label %cleanup40

cleanup40:                                        ; preds = %if.then12
  br label %return

return:                                           ; preds = %cleanup40
  ret void
}

; Make sure we don't try to outline the entire function.
; CHECK-LABEL: @fun
; CHECK-NOT: codeRepl
define void @fun() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %entry
  ret void
}

; Don't outline infinite loops.
; CHECK-LABEL: @infinite_loop
; CHECK-NOT: codeRepl
define void @infinite_loop() {
entry:
  br label %loop

loop:
  call void @sink()
  br label %loop
}

declare void @sink() cold
