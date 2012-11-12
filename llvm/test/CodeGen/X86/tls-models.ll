; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu -relocation-model=pic | FileCheck -check-prefix=X64_PIC %s
; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu -relocation-model=pic | FileCheck -check-prefix=X32_PIC %s

; Darwin always uses the same model.
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin | FileCheck -check-prefix=DARWIN %s

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
  ; X64:     f1:
  ; X64:     external_gd@GOTTPOFF
  ; X32:     f1:
  ; X32:     external_gd@INDNTPOFF
  ; X64_PIC: f1:
  ; X64_PIC: external_gd@TLSGD
  ; X32_PIC: f1:
  ; X32_PIC: external_gd@TLSGD
  ; DARWIN:  f1:
  ; DARWIN:  _external_gd@TLVP
}

define i32* @f2() {
entry:
  ret i32* @internal_gd

  ; Non-PIC code can use local exec, PIC code can use local dynamic.
  ; X64:     f2:
  ; X64:     internal_gd@TPOFF
  ; X32:     f2:
  ; X32:     internal_gd@NTPOFF
  ; X64_PIC: f2:
  ; X64_PIC: internal_gd@TLSLD
  ; X32_PIC: f2:
  ; X32_PIC: internal_gd@TLSLDM
  ; DARWIN:  f2:
  ; DARWIN:  _internal_gd@TLVP
}


; ----- localdynamic specified -----

define i32* @f3() {
entry:
  ret i32* @external_ld

  ; Non-PIC code can use initial exec, PIC code use local dynamic as specified.
  ; X64:     f3:
  ; X64:     external_ld@GOTTPOFF
  ; X32:     f3:
  ; X32:     external_ld@INDNTPOFF
  ; X64_PIC: f3:
  ; X64_PIC: external_ld@TLSLD
  ; X32_PIC: f3:
  ; X32_PIC: external_ld@TLSLDM
  ; DARWIN:  f3:
  ; DARWIN:  _external_ld@TLVP
}

define i32* @f4() {
entry:
  ret i32* @internal_ld

  ; Non-PIC code can use local exec, PIC code can use local dynamic.
  ; X64:     f4:
  ; X64:     internal_ld@TPOFF
  ; X32:     f4:
  ; X32:     internal_ld@NTPOFF
  ; X64_PIC: f4:
  ; X64_PIC: internal_ld@TLSLD
  ; X32_PIC: f4:
  ; X32_PIC: internal_ld@TLSLDM
  ; DARWIN:  f4:
  ; DARWIN:  _internal_ld@TLVP
}


; ----- initialexec specified -----

define i32* @f5() {
entry:
  ret i32* @external_ie

  ; Non-PIC and PIC code will use initial exec as specified.
  ; X64:     f5:
  ; X64:     external_ie@GOTTPOFF
  ; X32:     f5:
  ; X32:     external_ie@INDNTPOFF
  ; X64_PIC: f5:
  ; X64_PIC: external_ie@GOTTPOFF
  ; X32_PIC: f5:
  ; X32_PIC: external_ie@GOTNTPOFF
  ; DARWIN:  f5:
  ; DARWIN:  _external_ie@TLVP
}

define i32* @f6() {
entry:
  ret i32* @internal_ie

  ; Non-PIC code can use local exec, PIC code use initial exec as specified.
  ; X64:     f6:
  ; X64:     internal_ie@TPOFF
  ; X32:     f6:
  ; X32:     internal_ie@NTPOFF
  ; X64_PIC: f6:
  ; X64_PIC: internal_ie@GOTTPOFF
  ; X32_PIC: f6:
  ; X32_PIC: internal_ie@GOTNTPOFF
  ; DARWIN:  f6:
  ; DARWIN:  _internal_ie@TLVP
}


; ----- localexec specified -----

define i32* @f7() {
entry:
  ret i32* @external_le

  ; Non-PIC and PIC code will use local exec as specified.
  ; X64:     f7:
  ; X64:     external_le@TPOFF
  ; X32:     f7:
  ; X32:     external_le@NTPOFF
  ; X64_PIC: f7:
  ; X64_PIC: external_le@TPOFF
  ; X32_PIC: f7:
  ; X32_PIC: external_le@NTPOFF
  ; DARWIN:  f7:
  ; DARWIN:  _external_le@TLVP
}

define i32* @f8() {
entry:
  ret i32* @internal_le

  ; Non-PIC and PIC code will use local exec as specified.
  ; X64:     f8:
  ; X64:     internal_le@TPOFF
  ; X32:     f8:
  ; X32:     internal_le@NTPOFF
  ; X64_PIC: f8:
  ; X64_PIC: internal_le@TPOFF
  ; X32_PIC: f8:
  ; X32_PIC: internal_le@NTPOFF
  ; DARWIN:  f8:
  ; DARWIN:  _internal_le@TLVP
}
