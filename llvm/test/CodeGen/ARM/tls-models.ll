; RUN: llc -march=arm -mtriple=arm-linux-gnueabi < %s | FileCheck -check-prefix=CHECK-NONPIC %s
; RUN: llc -march=arm -mtriple=arm-linux-gnueabi -relocation-model=pic < %s | FileCheck -check-prefix=CHECK-PIC %s


@external_gd = external thread_local global i32
@internal_gd = internal thread_local global i32 42

@external_ld = external thread_local(localdynamic) global i32
@internal_ld = internal thread_local(localdynamic) global i32 42

@external_ie = external thread_local(initialexec) global i32
@internal_ie = internal thread_local(initialexec) global i32 42

@external_le = external thread_local(localexec) global i32
@internal_le = internal thread_local(localexec) global i32 42

; ----- no model specified -----

define i32* @f1() {
entry:
  ret i32* @external_gd

  ; Non-PIC code can use initial-exec, PIC code has to use general dynamic.
  ; CHECK-NONPIC:   f1:
  ; CHECK-NONPIC:   external_gd(gottpoff)
  ; CHECK-PIC:      f1:
  ; CHECK-PIC:      external_gd(tlsgd)
}

define i32* @f2() {
entry:
  ret i32* @internal_gd

  ; Non-PIC code can use local exec, PIC code can use local dynamic,
  ; but that is not implemented, so falls back to general dynamic.
  ; CHECK-NONPIC:   f2:
  ; CHECK-NONPIC:   internal_gd(tpoff)
  ; CHECK-PIC:      f2:
  ; CHECK-PIC:      internal_gd(tlsgd)
}


; ----- localdynamic specified -----

define i32* @f3() {
entry:
  ret i32* @external_ld

  ; Non-PIC code can use initial exec, PIC should use local dynamic,
  ; but that is not implemented, so falls back to general dynamic.
  ; CHECK-NONPIC:   f3:
  ; CHECK-NONPIC:   external_ld(gottpoff)
  ; CHECK-PIC:      f3:
  ; CHECK-PIC:      external_ld(tlsgd)
}

define i32* @f4() {
entry:
  ret i32* @internal_ld

  ; Non-PIC code can use local exec, PIC code can use local dynamic,
  ; but that is not implemented, so it falls back to general dynamic.
  ; CHECK-NONPIC:   f4:
  ; CHECK-NONPIC:   internal_ld(tpoff)
  ; CHECK-PIC:      f4:
  ; CHECK-PIC:      internal_ld(tlsgd)
}


; ----- initialexec specified -----

define i32* @f5() {
entry:
  ret i32* @external_ie

  ; Non-PIC and PIC code will use initial exec as specified.
  ; CHECK-NONPIC:   f5:
  ; CHECK-NONPIC:   external_ie(gottpoff)
  ; CHECK-PIC:      f5:
  ; CHECK-PIC:      external_ie(gottpoff)
}

define i32* @f6() {
entry:
  ret i32* @internal_ie

  ; Non-PIC code can use local exec, PIC code use initial exec as specified.
  ; CHECK-NONPIC:   f6:
  ; CHECK-NONPIC:   internal_ie(tpoff)
  ; CHECK-PIC:      f6:
  ; CHECK-PIC:      internal_ie(gottpoff)
}


; ----- localexec specified -----

define i32* @f7() {
entry:
  ret i32* @external_le

  ; Non-PIC and PIC code will use local exec as specified.
  ; CHECK-NONPIC:   f7:
  ; CHECK-NONPIC:   external_le(tpoff)
  ; CHECK-PIC:      f7:
  ; CHECK-PIC:      external_le(tpoff)
}

define i32* @f8() {
entry:
  ret i32* @internal_le

  ; Non-PIC and PIC code will use local exec as specified.
  ; CHECK-NONPIC:   f8:
  ; CHECK-NONPIC:   internal_le(tpoff)
  ; CHECK-PIC:      f8:
  ; CHECK-PIC:      internal_le(tpoff)
}
