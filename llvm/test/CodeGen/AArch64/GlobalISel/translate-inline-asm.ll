; RUN: llc -mtriple=aarch64-darwin-ios13 -O0 -global-isel -stop-after=irtranslator -o - %s | FileCheck %s

; The update_mir_test_checks script doesn't seem to handle INLINE_ASM well. Write this manually.

define void @asm_simple_memory_clobber() {
  ; CHECK-LABEL: name: asm_simple_memory_clobber
  ; CHECK: INLINEASM &"", 25
  ; CHECK:  INLINEASM &"", 1
  call void asm sideeffect "", "~{memory}"(), !srcloc !0
  call void asm sideeffect "", ""(), !srcloc !0
  ret void
}

!0 = !{i32 70}
