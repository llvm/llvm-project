; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -verify-machineinstrs -ppc-asm-full-reg-names \
; RUN:   -ppc-vsr-nums-as-vr < %s | FileCheck %s
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -verify-machineinstrs -ppc-asm-full-reg-names \
; RUN:   -ppc-vsr-nums-as-vr < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64-unknown-unknown \
; RUN:   -verify-machineinstrs -ppc-asm-full-reg-names \
; RUN:   -ppc-vsr-nums-as-vr < %s | FileCheck %s


@_ZTIi = external constant ptr

; Function is marked as nounwind but it still throws with __cxa_throw and
; calls __cxa_call_unexpected.
; Need to make sure that we do not only have a debug frame.
; Function Attrs: noreturn nounwind
define void @_Z4funcv() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
entry:
  %exception = tail call ptr @__cxa_allocate_exception(i64 4)
  store i32 100, ptr %exception, align 16
  invoke void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %1 = extractvalue { ptr, i32 } %0, 0
  tail call void @__cxa_call_unexpected(ptr %1)
  unreachable

unreachable:                                      ; preds = %entry
  unreachable
; CHECK-LABEL: _Z4funcv
; CHECK-NOT: .debug_frame
; CHECK: .cfi_personality
; CHECK: .cfi_endproc
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_call_unexpected(ptr) local_unnamed_addr


attributes #0 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="ppc64le" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+htm,+power8-vector,+vsx,-power9-vector" "unsafe-fp-math"="false" "use-soft-float"="false" }

