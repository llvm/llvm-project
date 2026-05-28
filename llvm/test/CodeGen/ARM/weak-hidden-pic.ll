; RUN: llc < %s -mtriple=armv7-linux-gnueabi -relocation-model=pic | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -relocation-model=pic | FileCheck %s

; Hidden weak function with dso_local must still use GOT indirection
; in PIC mode on ARM. The assembler eagerly resolves PC-relative
; expressions like .long xxx-(.LPC+8) when both are in the same section,
; which prevents the linker from overriding the weak definition with
; a non-weak one from another object file.

define weak dso_local hidden void @weak_hidden_func() {
  ret void
}

; CHECK-LABEL: weak_hidden_func_addr:
; CHECK:       .long weak_hidden_func(GOT_PREL)
define i8* @weak_hidden_func_addr() {
  ret i8* bitcast (void()* @weak_hidden_func to i8*)
}
