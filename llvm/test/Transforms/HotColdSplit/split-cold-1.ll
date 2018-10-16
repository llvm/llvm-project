; RUN: opt -hotcoldsplit -S < %s | FileCheck %s
; RUN: opt -passes=hotcoldsplit -S < %s | FileCheck %s

; Check that the function is not split. Outlined function is called from a
; basic block named codeRepl.

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

; Check that the function is not split. We used to outline the full function.

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
