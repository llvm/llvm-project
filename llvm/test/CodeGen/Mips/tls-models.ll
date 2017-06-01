; RUN: llc -mtriple=mipsel-- -relocation-model=pic < %s | FileCheck -check-prefix=CHECK-PIC %s
; RUN: llc -mtriple=mipsel-- -relocation-model=static < %s | FileCheck -check-prefix=CHECK-NONPIC %s

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
  ; CHECK-NONPIC-LABEL:   f1:
  ; CHECK-NONPIC:   %gottprel
  ; CHECK-PIC-LABEL:      f1:
  ; CHECK-PIC:      %tlsgd
}

define i32* @f2() {
entry:
  ret i32* @internal_gd

  ; Non-PIC code can use local exec, PIC code can use local dynamic.
  ; CHECK-NONPIC-LABEL:   f2:
  ; CHECK-NONPIC:   %tprel_hi
  ; CHECK-PIC-LABEL:      f2:
  ; CHECK-PIC:      %tlsldm
}


; ----- localdynamic specified -----

define i32* @f3() {
entry:
  ret i32* @external_ld

  ; Non-PIC code can use initial exec, PIC should use local dynamic.
  ; CHECK-NONPIC-LABEL:   f3:
  ; CHECK-NONPIC:   %gottprel
  ; CHECK-PIC-LABEL:      f3:
  ; CHECK-PIC:      %tlsldm
}

define i32* @f4() {
entry:
  ret i32* @internal_ld

  ; Non-PIC code can use local exec, PIC code can use local dynamic.
  ; CHECK-NONPIC-LABEL:   f4:
  ; CHECK-NONPIC:   %tprel_hi
  ; CHECK-PIC-LABEL:      f4:
  ; CHECK-PIC:      %tlsldm
}


; ----- initialexec specified -----

define i32* @f5() {
entry:
  ret i32* @external_ie

  ; Non-PIC and PIC code will use initial exec as specified.
  ; CHECK-NONPIC-LABEL:   f5:
  ; CHECK-NONPIC:   %gottprel
  ; CHECK-PIC-LABEL:      f5:
  ; CHECK-PIC:      %gottprel
}

define i32* @f6() {
entry:
  ret i32* @internal_ie

  ; Non-PIC code can use local exec, PIC code use initial exec as specified.
  ; CHECK-NONPIC-LABEL:   f6:
  ; CHECK-NONPIC:   %tprel_hi
  ; CHECK-PIC-LABEL:      f6:
  ; CHECK-PIC:      %gottprel
}


; ----- localexec specified -----

define i32* @f7() {
entry:
  ret i32* @external_le

  ; Non-PIC and PIC code will use local exec as specified.
  ; CHECK-NONPIC-LABEL:   f7:
  ; CHECK-NONPIC:   %tprel_hi
  ; CHECK-PIC-LABEL:      f7:
  ; CHECK-PIC:      %tprel_hi
}

define i32* @f8() {
entry:
  ret i32* @internal_le

  ; Non-PIC and PIC code will use local exec as specified.
  ; CHECK-NONPIC-LABEL:   f8:
  ; CHECK-NONPIC:   %tprel_hi
  ; CHECK-PIC-LABEL:      f8:
  ; CHECK-PIC:      %tprel_hi
}
