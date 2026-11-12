; RUN: opt < %s -O2 -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str.2 = private unnamed_addr constant [7 x i8] c"Boom!\0A\00", align 1

define dso_local void @trap() {
entry:
  unreachable
}

define dso_local void @test() personality ptr @__gxx_personality_v0 {
entry:
; CHECK: define dso_local void @test()
; CHECK-NEXT: entry:
; CHECK-NEXT: tail call void asm sideeffect
; CHECK-NEXT: ret void

  invoke void asm sideeffect "call trap", "~{dirflag},~{fpsr},~{flags}"()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
  call void (ptr, ...) @printf(ptr @.str.2)
  resume { ptr, i32 } %0

}

declare dso_local i32 @__gxx_personality_v0(...)

declare dso_local void @printf(ptr, ...)
