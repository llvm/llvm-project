; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu -relocation-model=pic | FileCheck -check-prefix=X64_PIC %s
; RUN: llc < %s -mtriple=i386-linux-gnu | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -mtriple=i386-linux-gnu -relocation-model=pic | FileCheck -check-prefix=X32_PIC %s

; Darwin always uses the same model.
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck -check-prefix=DARWIN %s

@external_gd = external thread_local global i32
@internal_gd = internal thread_local global i32 42

@external_ld = external thread_local(localdynamic) global i32
@internal_ld = internal thread_local(localdynamic) global i32 42

@external_ie = external thread_local(initialexec) global i32
@internal_ie = internal thread_local(initialexec) global i32 42

@external_le = external thread_local(localexec) global i32
@internal_le = internal thread_local(localexec) global i32 42

; See test cases for emulated model in emutls.ll, emutls-pic.ll and emutls-pie.ll.

; ----- no model specified -----

define ptr @f1() {
entry:
  ret ptr @external_gd

  ; Non-PIC code can use initial-exec, PIC code has to use general dynamic.
  ; X64-LABEL:     f1:
  ; X64:     external_gd@GOTTPOFF
  ; X32-LABEL:     f1:
  ; X32:     external_gd@INDNTPOFF
  ; X64_PIC-LABEL: f1:
  ; X64_PIC: external_gd@TLSGD
  ; X32_PIC-LABEL: f1:
  ; X32_PIC: external_gd@TLSGD
  ; DARWIN-LABEL:  f1:
  ; DARWIN:  _external_gd@TLVP
}

define ptr @f2() {
entry:
  ret ptr @internal_gd

  ; Non-PIC code can use local exec, PIC code can use local dynamic.
  ; X64-LABEL:     f2:
  ; X64:     internal_gd@TPOFF
  ; X32-LABEL:     f2:
  ; X32:     internal_gd@NTPOFF
  ; X64_PIC-LABEL: f2:
  ; X64_PIC: internal_gd@TLSLD
  ; X32_PIC-LABEL: f2:
  ; X32_PIC: internal_gd@TLSLDM
  ; DARWIN-LABEL:  f2:
  ; DARWIN:  _internal_gd@TLVP
}


; ----- localdynamic specified -----

define ptr @f3() {
entry:
  ret ptr @external_ld

  ; Non-PIC code can use initial exec, PIC code use local dynamic as specified.
  ; X64-LABEL:     f3:
  ; X64:     external_ld@GOTTPOFF
  ; X32-LABEL:     f3:
  ; X32:     external_ld@INDNTPOFF
  ; X64_PIC-LABEL: f3:
  ; X64_PIC: external_ld@TLSLD
  ; X32_PIC-LABEL: f3:
  ; X32_PIC: external_ld@TLSLDM
  ; DARWIN-LABEL:  f3:
  ; DARWIN:  _external_ld@TLVP
}

define ptr @f4() {
entry:
  ret ptr @internal_ld

  ; Non-PIC code can use local exec, PIC code can use local dynamic.
  ; X64-LABEL:     f4:
  ; X64:     internal_ld@TPOFF
  ; X32-LABEL:     f4:
  ; X32:     internal_ld@NTPOFF
  ; X64_PIC-LABEL: f4:
  ; X64_PIC: internal_ld@TLSLD
  ; X32_PIC-LABEL: f4:
  ; X32_PIC: internal_ld@TLSLDM
  ; DARWIN-LABEL:  f4:
  ; DARWIN:  _internal_ld@TLVP
}


; ----- initialexec specified -----

define ptr @f5() {
entry:
  ret ptr @external_ie

  ; Non-PIC and PIC code will use initial exec as specified.
  ; X64-LABEL:     f5:
  ; X64:     external_ie@GOTTPOFF
  ; X32-LABEL:     f5:
  ; X32:     external_ie@INDNTPOFF
  ; X64_PIC-LABEL: f5:
  ; X64_PIC: external_ie@GOTTPOFF
  ; X32_PIC-LABEL: f5:
  ; X32_PIC: external_ie@GOTNTPOFF
  ; DARWIN-LABEL:  f5:
  ; DARWIN:  _external_ie@TLVP
}

define ptr @f6() {
entry:
  ret ptr @internal_ie

  ; Non-PIC code can use local exec, PIC code use initial exec as specified.
  ; X64-LABEL:     f6:
  ; X64:     internal_ie@TPOFF
  ; X32-LABEL:     f6:
  ; X32:     internal_ie@NTPOFF
  ; X64_PIC-LABEL: f6:
  ; X64_PIC: internal_ie@GOTTPOFF
  ; X32_PIC-LABEL: f6:
  ; X32_PIC: internal_ie@GOTNTPOFF
  ; DARWIN-LABEL:  f6:
  ; DARWIN:  _internal_ie@TLVP
}

define i32 @PR22083() {
entry:
  ret i32 ptrtoint (ptr @external_ie to i32)
  ; X64-LABEL:     PR22083:
  ; X64:     movq    external_ie@GOTTPOFF(%rip), %rax
  ; X64_PIC-LABEL: PR22083:
  ; X64_PIC: movq    external_ie@GOTTPOFF(%rip), %rax
}

; ----- localexec specified -----

define ptr @f7() {
entry:
  ret ptr @external_le

  ; Non-PIC and PIC code will use local exec as specified.
  ; X64-LABEL:     f7:
  ; X64:     external_le@TPOFF
  ; X32-LABEL:     f7:
  ; X32:     external_le@NTPOFF
  ; X64_PIC-LABEL: f7:
  ; X64_PIC: external_le@TPOFF
  ; X32_PIC-LABEL: f7:
  ; X32_PIC: external_le@NTPOFF
  ; DARWIN-LABEL:  f7:
  ; DARWIN:  _external_le@TLVP
}

define ptr @f8() {
entry:
  ret ptr @internal_le

  ; Non-PIC and PIC code will use local exec as specified.
  ; X64-LABEL:     f8:
  ; X64:     internal_le@TPOFF
  ; X32-LABEL:     f8:
  ; X32:     internal_le@NTPOFF
  ; X64_PIC-LABEL: f8:
  ; X64_PIC: internal_le@TPOFF
  ; X32_PIC-LABEL: f8:
  ; X32_PIC: internal_le@NTPOFF
  ; DARWIN-LABEL:  f8:
  ; DARWIN:  _internal_le@TLVP
}
