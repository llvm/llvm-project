; RUN: llc -mtriple=arm64-apple-macosx -mattr=+mte %s -filetype=obj -o %t.o
; RUN: llvm-objdump --unwind-info %t.o | FileCheck %s

; Frames with MTE stack tagging must use DWARF unwinding because compact unwind
; doesn't handle MTE tag untagging during exception unwinding.

; MTE-tagged frame should use DWARF mode (0x03000000)
; CHECK-LABEL: Contents of __compact_unwind section:
; CHECK:       compact encoding: 0x03000000

; Normal frame should NOT use DWARF mode
; CHECK-NOT:   compact encoding: 0x03000000
; CHECK:       compact encoding: 0x{{[0-9a-f]+}}

define void @mte_tagged_frame() sanitize_memtag "frame-pointer"="all" {
  %x = alloca i32, align 4
  store i32 42, ptr %x
  call void asm sideeffect "", "r"(ptr %x)
  ret void
}

define void @normal_frame() "frame-pointer"="all" {
  %x = alloca i32, align 4
  store i32 42, ptr %x
  call void asm sideeffect "", "r"(ptr %x)
  ret void
}
