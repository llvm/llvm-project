; RUN: llc -mtriple=mips -relocation-model=static < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -relocation-model=static < %s | FileCheck %s
; RUN: llc -mtriple=mips64el -relocation-model=static < %s | FileCheck %s

; %a on a register operand prints it as a memory address with a zero
; offset, as GCC does. It used to print the next INLINEASM operand as
; the offset.

define i32 @load(ptr %p) {
; CHECK-LABEL: load:
; CHECK:       #APP
; CHECK:       lw $2, 0($4)
; CHECK:       #NO_APP
entry:
  %r = tail call i32 asm sideeffect "lw $0, ${1:a}", "=r,r,~{memory},~{$1}"(ptr %p)
  ret i32 %r
}

define i32 @load_no_clobber(ptr %p) {
; CHECK-LABEL: load_no_clobber:
; CHECK:       #APP
; CHECK:       lw $2, 0($4)
; CHECK:       #NO_APP
entry:
  %r = tail call i32 asm sideeffect "lw $0, ${1:a}", "=r,r"(ptr %p)
  ret i32 %r
}
