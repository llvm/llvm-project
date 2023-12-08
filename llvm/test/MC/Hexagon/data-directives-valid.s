# RUN: llvm-mc -filetype=obj -triple hexagon < %s \
# RUN:     | llvm-objdump -s - | FileCheck %s

.data

# CHECK: Contents of section .data:
# CHECK-NEXT: 0000 deadbeef badcaf11 22334455 66778800
.byte 0xde
.half 0xbead
.word 0xafdcbaef
.8byte 0x8877665544332211
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
