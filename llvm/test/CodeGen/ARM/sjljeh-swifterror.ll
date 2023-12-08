; RUN: opt -sjljehprepare -verify < %s -S | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv7s-apple-ios7.0"

%swift.error = type opaque

declare void @objc_msgSend() local_unnamed_addr

declare i32 @__objc_personality_v0(...)

; Make sure we don't leave a select on a swifterror argument.
; CHECK-LABEL: @test
; CHECK-NOT: select true, %0
define swiftcc void @test(ptr swifterror) local_unnamed_addr personality ptr @__objc_personality_v0 {
entry:
  %call28.i = invoke i32 @objc_msgSend(ptr undef, ptr undef)
          to label %invoke.cont.i unwind label %lpad.i

invoke.cont.i:
  unreachable

lpad.i:
  %1 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef
}

%struct._objc_typeinfo = type { ptr, ptr, ptr }
@"OBJC_EHTYPE_$_NSException" = external global %struct._objc_typeinfo

; Make sure this does not crash.
; CHECK-LABEL: @swift_error_bug
; CHECK: store ptr null, ptr %0

define hidden swiftcc void @swift_error_bug(ptr swifterror, ptr %fun, i1 %b) local_unnamed_addr #0 personality ptr @__objc_personality_v0 {
  %2 = load ptr, ptr %fun, align 4
  invoke void %2(ptr null) #1
          to label %tryBlock.exit unwind label %3, !clang.arc.no_objc_arc_exceptions !1

; <label>:3:
  %4 = landingpad { ptr, i32 }
          catch ptr @"OBJC_EHTYPE_$_NSException"
  br label %tryBlock.exit

tryBlock.exit:
  br i1 %b, label %5, label %_T0ypMa.exit.i.i

_T0ypMa.exit.i.i:
  store ptr null, ptr %0, align 4
  ret void

; <label>:5:
  ret void
}

!1 = !{}
