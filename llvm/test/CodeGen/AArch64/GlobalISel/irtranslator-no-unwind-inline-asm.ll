; RUN: llc -O0 -global-isel -stop-after=irtranslator < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@.str.2 = private unnamed_addr constant [7 x i8] c"Boom!\0A\00", align 1

define dso_local void @trap() {
entry:
  unreachable
}

define dso_local void @test() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:

; CHECK-LABEL: name: test
; CHECK: body:
; CHECK-NEXT: bb.1.entry
; CHECK-NOT: EH_LABEL
; CHECK: INLINEASM
; CHECK-NOT: EH_LABEL

  invoke void asm sideeffect "bl trap", ""()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
; CHECK: bb.3.lpad
; CHECK: EH_LABEL

  %0 = landingpad { i8*, i32 }
          cleanup
  call void (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.2, i64 0, i64 0))
  resume { i8*, i32 } %0

}

declare dso_local i32 @__gxx_personality_v0(...)

declare dso_local void @printf(i8*, ...)
