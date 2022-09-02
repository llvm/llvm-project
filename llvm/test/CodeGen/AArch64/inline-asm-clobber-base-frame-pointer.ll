; Check that not only do we warn about clobbering x19 we also say
; what it is used for.

; RUN: llc <%s -mtriple=aarch64-none-eabi 2>&1 | FileCheck %s

; CHECK: warning: inline asm clobber list contains reserved registers: X19
; CHECK: note: X19 is used as the frame base pointer register.

define void @alloca(i64 %size) {
entry:
  %a = alloca i128, i64 %size, align 64
  call void asm sideeffect "nop", "~{x19}"()
  ret void
}

