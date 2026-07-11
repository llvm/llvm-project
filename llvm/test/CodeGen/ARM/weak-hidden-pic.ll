; RUN: llc < %s -mtriple=armv7-linux-gnueabi -relocation-model=pic | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -relocation-model=pic | FileCheck %s
; RUN: llc < %s -O0 -fast-isel-abort=2 -mtriple=armv7-linux-gnueabi -relocation-model=pic | FileCheck %s

; Weak dso_local hidden functions must be overridable at link time
; (a non-weak definition in another object should override the weak one).
; Instead of using GOT indirection, we use a PC-relative constant pool
; entry with a .reloc directive to force the assembler to emit a proper
; relocation (R_ARM_REL32), preventing eager resolution when the symbol
; and reference are in the same section.

define weak dso_local hidden void @weak_hidden_func() {
  ret void
}

; CHECK-LABEL: weak_hidden_func_addr:
; CHECK:       .long .Ltmp{{[0-9]+}}-(.LPC{{[0-9]+}}_0+{{[48]}})
; CHECK:       .reloc .Ltmp{{[0-9]+}}, R_ARM_REL32, weak_hidden_func
define i8* @weak_hidden_func_addr() {
  ret i8* bitcast (void()* @weak_hidden_func to i8*)
}
