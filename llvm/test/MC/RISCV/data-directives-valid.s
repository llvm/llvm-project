# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -s - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -s - | FileCheck %s

# Check that data directives supported by gas are also supported by LLVM MC.
# As there was some confusion about whether .half/.word/.dword imply
# alignment (see <https://github.com/riscv/riscv-asm-manual/issues/12>), we
# are sure to check this.

.data

# CHECK: Contents of section .data:
# CHECK-NEXT: 0000 deadbeef badcaf11 22334455 66778800
.byte 0xde
.half 0xbead
.word 0xafdcbaef
.dword 0x8877665544332211
.byte 0

# CHECK-NEXT: 0010 deadbeef badcaf11 22334455 66778800
.byte 0xde
.2byte 0xbead
.4byte 0xafdcbaef
.8byte 0x8877665544332211
.byte 0

# CHECK-NEXT: 0020 deadbeef badcaf11 22
.byte 0xde
.short 0xbead
.long 0xafdcbaef
.hword 0x2211
