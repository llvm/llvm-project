; RUN: opt < %s -simplifycfg -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare i32 @__gxx_personality_v0(...)
declare void @fn()


; CHECK-LABEL: @test1
define void @test1() personality i32 (...)* @__gxx_personality_v0 {
entry:
; CHECK-LABEL: entry:
; CHECK: to label %invoke2 unwind label %lpad2
  invoke void @fn()
    to label %invoke2 unwind label %lpad1

invoke2:
; CHECK-LABEL: invoke2:
; CHECK: to label %invoke.cont unwind label %lpad2
  invoke void @fn()
    to label %invoke.cont unwind label %lpad2

invoke.cont:
  ret void

lpad1:
  %exn = landingpad {i8*, i32}
         cleanup
  br label %shared_resume

lpad2:
; CHECK-LABEL: lpad2:
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: call void @fn()
; CHECK-NEXT: ret void
  %exn2 = landingpad {i8*, i32}
          cleanup
  br label %shared_resume

shared_resume:
  call void @fn()
  ret void
}

; Don't trigger if blocks aren't the same/empty
define void @neg1() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: @neg1
entry:
; CHECK-LABEL: entry:
; CHECK: to label %invoke2 unwind label %lpad1
  invoke void @fn()
    to label %invoke2 unwind label %lpad1

invoke2:
; CHECK-LABEL: invoke2:
; CHECK: to label %invoke.cont unwind label %lpad2
  invoke void @fn()
    to label %invoke.cont unwind label %lpad2

invoke.cont:
  ret void

lpad1:
  %exn = landingpad {i8*, i32}
         filter [0 x i8*] zeroinitializer
  call void @fn()
  br label %shared_resume

lpad2:
  %exn2 = landingpad {i8*, i32}
          cleanup
  br label %shared_resume

shared_resume:
  call void @fn()
  ret void
}

; Should not trigger when the landing pads are not the exact same
define void @neg2() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: @neg2
entry:
; CHECK-LABEL: entry:
; CHECK: to label %invoke2 unwind label %lpad1
  invoke void @fn()
    to label %invoke2 unwind label %lpad1

invoke2:
; CHECK-LABEL: invoke2:
; CHECK: to label %invoke.cont unwind label %lpad2
  invoke void @fn()
    to label %invoke.cont unwind label %lpad2

invoke.cont:
  ret void

lpad1:
  %exn = landingpad {i8*, i32}
         filter [0 x i8*] zeroinitializer
  br label %shared_resume

lpad2:
  %exn2 = landingpad {i8*, i32}
          cleanup
  br label %shared_resume

shared_resume:
  call void @fn()
  ret void
}
