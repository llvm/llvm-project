// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux-gnu %s -o - \
// RUN:   | llvm-readobj -s -t | FileCheck --check-prefix=OBJ %s

// RUN: llvm-mc -filetype=asm -triple=x86_64-pc-linux-gnu %s -o - \
// RUN:   | FileCheck --check-prefix=ASM %s

  .section .sec1,"a",@0x70000001
  .section .sec2,"a",@1879048193

// OBJ:      Section {
// OBJ:        Name: .sec1
// OBJ-NEXT:   Type: SHT_X86_64_UNWIND (0x70000001)
// OBJ:      }
// OBJ:      Section {
// OBJ:        Name: .sec2
// OBJ-NEXT:   Type: SHT_X86_64_UNWIND (0x70000001)
// OBJ:      }

// ASM: .section  .sec1,"a",@unwind
// ASM: .section  .sec2,"a",@unwind
