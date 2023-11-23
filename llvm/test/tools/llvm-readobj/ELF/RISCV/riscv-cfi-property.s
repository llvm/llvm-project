# RUN: rm -rf %t && split-file %s %t && cd %t

#--- gnu-property-riscv32.s
// RUN: llvm-mc -triple riscv32 -filetype obj gnu-property-riscv32.s -o gnu-property-riscv32.o
// RUN: llvm-readobj -n gnu-property-riscv32.o | FileCheck -check-prefix=LLVM gnu-property-riscv32.s
// RUN: llvm-readobj -n --elf-output-style=GNU gnu-property-riscv32.o | FileCheck -check-prefix=GNU gnu-property-riscv32.s


// LLVM:      Notes [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.gnu.property
// LLVM-NEXT:     Offset: 0x34
// LLVM-NEXT:     Size: 0x1C
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: GNU
// LLVM-NEXT:       Data size: 0xC
// LLVM-NEXT:       Type: NT_GNU_PROPERTY_TYPE_0 (property note)
// LLVM-NEXT:       Property [
// LLVM-NEXT:         riscv feature: Zicfilp, Zicfiss
// LLVM-NEXT:       ]
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT: ]

// GNU:       Displaying notes found in: .note.gnu.property
// GNU-NEXT:  Owner                Data size        Description
// GNU-NEXT:  GNU                  0x0000000c       NT_GNU_PROPERTY_TYPE_0 (property note)
// GNU-NEXT:    Properties:    riscv feature: Zicfilp, Zicfiss

// GNU Note Section Example
.section .note.gnu.property, "a"
  .p2align 2
  .long 4
  .long 12;
  .long 5
  .asciz "GNU"
  .long 0xc0000000
  .long 4
  .long 3

#--- gnu-property-riscv64.s
// RUN: llvm-mc -triple riscv64 -filetype obj gnu-property-riscv64.s -o gnu-property-riscv64.o
// RUN: llvm-readobj -n gnu-property-riscv64.o | FileCheck -check-prefix=LLVM64 gnu-property-riscv64.s
// RUN: llvm-readobj -n --elf-output-style=GNU gnu-property-riscv64.o | FileCheck -check-prefix=GNU64 gnu-property-riscv64.s


// LLVM64:      Notes [
// LLVM64-NEXT:   NoteSection {
// LLVM64-NEXT:     Name: .note.gnu.property
// LLVM64-NEXT:     Offset: 0x40
// LLVM64-NEXT:     Size: 0x20
// LLVM64-NEXT:     Note {
// LLVM64-NEXT:       Owner: GNU
// LLVM64-NEXT:       Data size: 0x10
// LLVM64-NEXT:       Type: NT_GNU_PROPERTY_TYPE_0 (property note)
// LLVM64-NEXT:       Property [
// LLVM64-NEXT:         riscv feature: Zicfilp, Zicfiss
// LLVM64-NEXT:       ]
// LLVM64-NEXT:     }
// LLVM64-NEXT:   }
// LLVM64-NEXT: ]

// GNU64:       Displaying notes found in: .note.gnu.property
// GNU64-NEXT:  Owner                Data size        Description
// GNU64-NEXT:  GNU                  0x00000010       NT_GNU_PROPERTY_TYPE_0 (property note)
// GNU64-NEXT:    Properties:    riscv feature: Zicfilp, Zicfiss

// GNU Note Section Example
.section .note.gnu.property, "a"
  .p2align 2
  .long 4
  .long 16;
  .long 5
  .asciz "GNU"
  .long 0xc0000000
  .long 4
  .long 3
  .long 0
