# RUN: llvm-mc -filetype=obj -triple=riscv32be %s -o %t.32be.o
# RUN: llvm-objdump -s %t.32be.o | FileCheck -check-prefix=RV32BE %s
# RUN: llvm-mc -filetype=obj -triple=riscv64be %s -o %t.64be.o
# RUN: llvm-objdump -s %t.64be.o | FileCheck -check-prefix=RV64BE %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32le.o
# RUN: llvm-objdump -s %t.32le.o | FileCheck -check-prefix=RV32LE %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.64le.o
# RUN: llvm-objdump -s %t.64le.o | FileCheck -check-prefix=RV64LE %s

# Test that data directives are properly byte-swapped on big-endian RISC-V

.data

byte_data:
  .byte 0x11
  .byte 0x22
  .half 0x3344
  .word 0x55667788
  .long 0x99aabbcc
  .quad 0x1122334455667788

# RV32BE: Contents of section .data:
# RV32BE-NEXT:  0000 11223344 55667788 99aabbcc 11223344
# RV32BE-NEXT:  0010 55667788

# RV64BE: Contents of section .data:
# RV64BE-NEXT:  0000 11223344 55667788 99aabbcc 11223344
# RV64BE-NEXT:  0010 55667788

# RV32LE: Contents of section .data:
# RV32LE-NEXT:  0000 11224433 88776655 ccbbaa99 88776655
# RV32LE-NEXT:  0010 44332211

# RV64LE: Contents of section .data:
# RV64LE-NEXT:  0000 11224433 88776655 ccbbaa99 88776655
# RV64LE-NEXT:  0010 44332211
