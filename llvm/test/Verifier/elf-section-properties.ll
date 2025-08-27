; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: elf_section_properties metadata must have two operands
@g1 = global i32 0, !elf_section_properties !{i32 0}
; CHECK: type field must be ConstantAsMetadata
@g2 = global i32 0, !elf_section_properties !{!{}, i32 0}
; CHECK: entsize field must be ConstantAsMetadata
@g3 = global i32 0, !elf_section_properties !{i32 0, !{}}
; CHECK: type field must be ConstantInt
@g4 = global i32 0, !elf_section_properties !{float 0.0, i32 0}
; CHECK: entsize field must be ConstantInt
@g5 = global i32 0, !elf_section_properties !{i32 0, float 0.0}
