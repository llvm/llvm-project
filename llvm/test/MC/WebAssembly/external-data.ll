; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s
; Verify relocations are correctly generated for addresses of externals
; in the data section.

@myimport = external global i32, align 4

@foo = global i64 7, align 4
@bar = hidden global i32* @myimport, align 4

; CHECK:   - Type:            DATA
; CHECK:     Relocations:
; CHECK:       - Type:            R_WEBASSEMBLY_GLOBAL_ADDR_I32
; CHECK:         Index:           0
; CHECK:         Offset:          0x0000000E
; CHECK:     Segments:
; CHECK:       - SectionOffset:   6
; CHECK:         MemoryIndex:     0
; CHECK:         Offset:
; CHECK:           Opcode:          I32_CONST
; CHECK:           Value:           0
; CHECK:         Content:         0700000000000000FFFFFFFF

