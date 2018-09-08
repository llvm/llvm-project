; RUN: opt -hotcoldsplit -S < %s | FileCheck %s
source_filename = "bugpoint-output-054409e.bc"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

declare i32 @__gxx_personality_v0(...)

; Outlined function is called from a basic block named codeRepl
; CHECK: codeRepl:
; CHECK-NEXT: call coldcc void @foo
; Check that no recursive outlining is done.
; CHECK-NOT: codeRepl:
; Function Attrs: ssp uwtable
define hidden void @foo() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
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


