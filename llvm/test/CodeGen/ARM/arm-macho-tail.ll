; RUN: llc -mtriple=thumbv7-apple-watchos %s -filetype=obj -o %t
; RUN: llvm-objdump -r %t | FileCheck %s

  ; Relocation needs to explicitly mention _bar rather than be __text relative
  ; because the __text relative offset is not encodable in an ARM instruction.
; CHECK: ARM_RELOC_BR24 _bar
define void @foo() "target-features"="-thumb-mode" {
  tail call void @bar()
  ret void
}

define void @one_inst() { ret void }

define void @bar() {
  ret void
}
