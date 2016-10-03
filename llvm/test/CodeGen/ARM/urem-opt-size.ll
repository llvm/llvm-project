; When optimising for minimum size, we don't want to expand a div to a mul
; and a shift sequence. As a result, the urem instruction e.g. will not be
; expanded to a sequence of umull, lsrs, muls and sub instructions, but
; just a call to __aeabi_uidivmod.
;
; When the processor features hardware division, UDIV + UREM can be turned
; into UDIV + MLS. This prevents the library function __aeabi_uidivmod to be
; pulled into the binary. The test uses ARMv7-M.
;
; RUN: llc -mtriple=armv7a-eabi -mattr=-neon -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7m-eabi -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=V7M

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

define i32 @foo1() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: foo1:
; CHECK:__aeabi_idiv
; CHECK-NOT: smmul
  %call = tail call i32 bitcast (i32 (...)* @GetValue to i32 ()*)()
  %div = sdiv i32 %call, 1000000
  ret i32 %div
}

define i32 @foo2() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: foo2:
; CHECK: __aeabi_uidiv
; CHECK-NOT: umull
  %call = tail call i32 bitcast (i32 (...)* @GetValue to i32 ()*)()
  %div = udiv i32 %call, 1000000
  ret i32 %div
}

; Test for unsigned remainder
define i32 @foo3() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: foo3:
; CHECK: __aeabi_uidivmod
; CHECK-NOT: umull
; V7M-LABEL: foo3:
; V7M: udiv [[R2:r[0-9]+]], [[R0:r[0-9]+]], [[R1:r[0-9]+]]
; V7M: mls {{r[0-9]+}}, [[R2]], [[R1]], [[R0]]
; V7M-NOT: __aeabi_uidivmod
  %call = tail call i32 bitcast (i32 (...)* @GetValue to i32 ()*)()
  %rem = urem i32 %call, 1000000
  %cmp = icmp eq i32 %rem, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Test for signed remainder
define i32 @foo4() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: foo4:
; CHECK:__aeabi_idivmod
; V7M-LABEL: foo4:
; V7M: sdiv [[R2:r[0-9]+]], [[R0:r[0-9]+]], [[R1:r[0-9]+]]
; V7M: mls {{r[0-9]+}}, [[R2]], [[R1]], [[R0]]
; V7M-NOT: __aeabi_idivmod
  %call = tail call i32 bitcast (i32 (...)* @GetValue to i32 ()*)()
  %rem = srem i32 %call, 1000000
  ret i32 %rem
}

; Check that doing a sdiv+srem has the same effect as only the srem,
; as the division needs to be computed anyway in order to calculate
; the remainder (i.e. make sure we don't end up with two divisions).
define i32 @foo5() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: foo5:
; CHECK:__aeabi_idivmod
; V7M-LABEL: foo5:
; V7M: sdiv [[R2:r[0-9]+]], [[R0:r[0-9]+]], [[R1:r[0-9]+]]
; V7M-NOT: sdiv
; V7M: mls {{r[0-9]+}}, [[R2]], [[R1]], [[R0]]
; V7M-NOT: __aeabi_idivmod
  %call = tail call i32 bitcast (i32 (...)* @GetValue to i32 ()*)()
  %div = sdiv i32 %call, 1000000
  %rem = srem i32 %call, 1000000
  %add = add i32 %div, %rem
  ret i32 %add
}

; An early version of this patch caused isel to hang. The reason
; was that it shouldn't do the rewrite for i64 because that's not
; supported by hardware. Isel was stuck in a loop with type
; legalization and this optimisation.
; Function Attrs: norecurse nounwind
define i64 @isel_dont_hang(i32 %bar) local_unnamed_addr #4 {
entry:
; CHECK-LABEL: isel_dont_hang:
; CHECK: __aeabi_uldivmod
  %temp.0 = sext i32 %bar to i64
  %mul83 = shl i64 %temp.0, 1
  %add84 = add i64 %temp.0, 2
  %div85 = udiv i64 %mul83, %add84
  ret i64 %div85
}

; i16 types are promoted to i32, and we expect a normal udiv here:
define i16 @isel_dont_hang_2(i16 %bar) local_unnamed_addr #4 {
entry:
; CHECK-LABEL: isel_dont_hang_2:
; CHECK: udiv
; CHECK-NOT: __aeabi_
  %mul83 = shl i16 %bar, 1
  %add84 = add i16 %bar, 2
  %div85 = udiv i16 %mul83, %add84
  ret i16 %div85
}
declare i32 @GetValue(...) local_unnamed_addr

attributes #0 = { minsize nounwind optsize }
attributes #4 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-jump-tables"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a15" "target-features"="+dsp,+hwdiv,+hwdiv-arm,+neon,+vfp4" "use-soft-float"="false" }

