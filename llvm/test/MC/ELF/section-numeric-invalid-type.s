// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux-gnu %s -o - \
// RUN:   | llvm-readobj -S --symbols | FileCheck --check-prefix=OBJ %s

// RUN: not llvm-mc -filetype=asm -triple=x86_64-pc-linux-gnu %s -o - 2>&1 \
// RUN:   | FileCheck --check-prefix=ASM %s

  .section .sec,"a",@0x7fffffff

// OBJ:      Section {
// OBJ:        Name: .sec
// OBJ-NEXT:   Type: Unknown (0x7FFFFFFF)
// OBJ:      }

// ASM: unsupported type 0x7fffffff for section .sec
