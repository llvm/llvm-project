# Check that .rdata sections have proper name, flags, and section types.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o - \
# RUN:   | llvm-readobj -S - | FileCheck %s

  .rdata
  .word 0

# CHECK:      Name: .rodata
# CHECK-NEXT: Type: SHT_PROGBITS
# CHECK-NEXT: Flags [ (0x2)
# CHECK-NEXT:   SHF_ALLOC
# CHECK-NEXT: ]
