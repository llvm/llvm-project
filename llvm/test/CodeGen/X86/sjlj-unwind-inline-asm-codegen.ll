; FIXME: Fix machine verifier issues and remove -verify-machineinstrs=0. PR39439.
; RUN: llc --exception-model=sjlj -verify-machineinstrs=0 < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str.2 = private unnamed_addr constant [7 x i8] c"Boom!\0A\00", align 1

define dso_local void @trap() {
entry:
  unreachable
}

define dso_local void @test() personality ptr @__gxx_personality_sj0 {
entry:

; CHECK: callq  _Unwind_SjLj_Register@PLT
; CHECK-LABEL: .Ltmp0:
; CHECK: callq  trap
; CHECK-LABEL: .Ltmp1:
; CHECK: callq  _Unwind_SjLj_Unregister@PLT

  invoke void asm sideeffect unwind "call trap", "~{dirflag},~{fpsr},~{flags}"()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
; CHECK: callq  printf
; CHECK: callq  _Unwind_SjLj_Resume@PLT
  call void (ptr, ...) @printf(ptr @.str.2)
  resume { ptr, i32 } %0

}

declare dso_local i32 @__gxx_personality_sj0(...)

declare dso_local void @printf(ptr, ...)
