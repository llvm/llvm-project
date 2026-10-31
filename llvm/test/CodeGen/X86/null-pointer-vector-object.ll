; RUN: llc -mtriple=x86_64-unknown-linux-gnu -data-sections -filetype=obj %s -o %t.64
; RUN: llvm-readobj --symbols %t.64 | FileCheck %s --check-prefixes=CHECK,ELF64
; RUN: llc -mtriple=i686-unknown-linux-gnu -data-sections -filetype=obj %s -o %t.32
; RUN: llvm-readobj --symbols %t.32 | FileCheck %s --check-prefixes=CHECK,ELF32
; RUN: llc -mtriple=x86_64-pc-windows-msvc -data-sections -filetype=obj %s -o %t.obj
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -data-sections %s -o - | FileCheck %s --check-prefix=ASM64
; RUN: llc -mtriple=i686-unknown-linux-gnu -data-sections %s -o - | FileCheck %s --check-prefix=ASM32

; Both spellings of a null pointer vector use ConstantPointerNull. Its size
; can exceed the eight-byte limit of MCStreamer::emitIntValue. Exercise object
; emission directly, including scalar and array controls.
@scalar = global ptr null
@v2 = global <2 x ptr> zeroinitializer
@v4 = global <4 x ptr> splat (ptr null)
@array = global [2 x ptr] zeroinitializer

; Scalar null pointers should retain their existing assembly representation.
; ASM64-LABEL: scalar:
; ASM64-NEXT:  .quad 0
; ASM32-LABEL: scalar:
; ASM32-NEXT:  .long 0

; CHECK:      Name: scalar
; ELF64:      Size: 8
; ELF32:      Size: 4
; CHECK:      Section: .bss.scalar
; CHECK:      Name: v2
; ELF64:      Size: 16
; ELF32:      Size: 8
; CHECK:      Section: .bss.v2
; CHECK:      Name: v4
; ELF64:      Size: 32
; ELF32:      Size: 16
; CHECK:      Section: .bss.v4
; CHECK:      Name: array
; ELF64:      Size: 16
; ELF32:      Size: 8
; CHECK:      Section: .bss.array
