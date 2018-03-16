; RUN: opt -sjljehprepare -verify < %s -S | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv7s-apple-ios7.0"

%swift.error = type opaque

declare void @objc_msgSend() local_unnamed_addr

declare i32 @__objc_personality_v0(...)

; Make sure we don't leave a select on a swifterror argument.
; CHECK-LABEL: @test
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

%struct._objc_typeinfo = type { i8**, i8*, i64* }
@"OBJC_EHTYPE_$_NSException" = external global %struct._objc_typeinfo

; Make sure this does not crash.
; CHECK-LABEL: @swift_error_bug
; CHECK: store %swift.error* null, %swift.error** %0

define hidden swiftcc void @swift_error_bug(%swift.error** swifterror, void (i8*)** %fun, i1 %b) local_unnamed_addr #0 personality i32 (...)* @__objc_personality_v0 {
  %2 = load void (i8*)*, void (i8*)** %fun, align 4
  invoke void %2(i8* null) #1
          to label %tryBlock.exit unwind label %3, !clang.arc.no_objc_arc_exceptions !1

; <label>:3:
  %4 = landingpad { i8*, i32 }
          catch %struct._objc_typeinfo* @"OBJC_EHTYPE_$_NSException"
  br label %tryBlock.exit

tryBlock.exit:
  br i1 %b, label %5, label %_T0ypMa.exit.i.i

_T0ypMa.exit.i.i:
  store %swift.error* null, %swift.error** %0, align 4
  ret void

; <label>:5:
  ret void
}

!1 = !{}
