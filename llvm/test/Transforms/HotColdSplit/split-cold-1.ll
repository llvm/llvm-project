; RUN: opt -hotcoldsplit -S < %s | FileCheck %s

; Outlined function is called from a basic block named codeRepl
; CHECK: codeRepl:
; CHECK-NEXT: call void @foo
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
