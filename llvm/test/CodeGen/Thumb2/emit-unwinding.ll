; RUN: llc -mtriple thumbv7em-apple-unknown-eabi-macho %s -o - -O0 | FileCheck %s

; CHECK: add.w r11, sp, #{{[1-9]+}}

define void @foo1() {
  call void asm sideeffect "", "~{r4}"()
  call void @foo2()
  ret void
}

declare void @foo2()
