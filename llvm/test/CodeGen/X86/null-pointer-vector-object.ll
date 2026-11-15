; RUN: llc -mtriple=x86_64-unknown-linux-gnu -data-sections -filetype=obj %s -o %t.64
; RUN: llvm-readobj --symbols %t.64 | FileCheck %s --check-prefixes=CHECK,ELF64
; RUN: llc -mtriple=i686-unknown-linux-gnu -data-sections -filetype=obj %s -o %t.32
; RUN: llvm-readobj --symbols %t.32 | FileCheck %s --check-prefixes=CHECK,ELF32
; RUN: llc -mtriple=x86_64-pc-windows-msvc -data-sections -filetype=obj %s -o /dev/null

; Both spellings of a null pointer vector use ConstantPointerNull. Its size
; can exceed the eight-byte limit of MCStreamer::emitIntValue. Check successful
; object emission for ELF and COFF, and symbol sizes and BSS placement for ELF.
@v2 = global <2 x ptr> zeroinitializer
@v4 = global <4 x ptr> splat (ptr null)

; CHECK:      Name: v2
; ELF64:      Size: 16
; ELF32:      Size: 8
; CHECK:      Section: .bss.v2
; CHECK:      Name: v4
; ELF64:      Size: 32
; ELF32:      Size: 16
; CHECK:      Section: .bss.v4
