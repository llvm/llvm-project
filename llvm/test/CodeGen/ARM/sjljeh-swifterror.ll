; RUN: opt -sjljehprepare -verify < %s | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv7s-apple-ios7.0"

%swift.error = type opaque

declare void @objc_msgSend() local_unnamed_addr

declare i32 @__objc_personality_v0(...)

; Make sure we don't leave a select on a swifterror argument.
; CHECK-LABEL; @test
; CHECK-NOT: select true, %0
define swiftcc void @test(%swift.error** swifterror) local_unnamed_addr personality i32 (...)* @__objc_personality_v0 {
entry:
  %call28.i = invoke i32 bitcast (void ()* @objc_msgSend to i32 (i8*, i8*)*)(i8* undef, i8* undef)
          to label %invoke.cont.i unwind label %lpad.i

invoke.cont.i:
  unreachable

lpad.i:
  %1 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

