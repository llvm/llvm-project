; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s


@_ZTIi = external constant i8*

; Function is marked as nounwind but it still throws with __cxa_throw and
; calls __cxa_call_unexpected.
; Need to make sure that we do not only have a debug frame.
; Function Attrs: noreturn nounwind
define void @_Z4funcv() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %exception = tail call i8* @__cxa_allocate_exception(i64 4)
  %0 = bitcast i8* %exception to i32*
  store i32 100, i32* %0, align 16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %2 = extractvalue { i8*, i32 } %1, 0
  tail call void @__cxa_call_unexpected(i8* %2)
  unreachable

unreachable:                                      ; preds = %entry
  unreachable
; CHECK-LABEL: _Z4funcv
; CHECK-NOT: .debug_frame
; CHECK: .cfi_personality
; CHECK: .cfi_endproc
}

declare i8* @__cxa_allocate_exception(i64) local_unnamed_addr

declare void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_call_unexpected(i8*) local_unnamed_addr


attributes #0 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

