# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck %s

# The format of the .eh_frame section is similar in
# format and purpose to the .debug_frame section.
# Version 1 is often used for .eh_frame,
# and also it was used for DWARF v2. For that case,
# return address register should be encoded as ubyte,
# while later versions use ULEB128. This test case
# checks that we are able to dump it correctly.

# CHECK:      .eh_frame contents:
# CHECK:      00000000 00000010 ffffffff CIE
# CHECK-NEXT:   Version:               1
# CHECK-NEXT:   Augmentation:          "zR"
# CHECK-NEXT:   Code alignment factor: 1
# CHECK-NEXT:   Data alignment factor: 1
# CHECK-NEXT:   Return address column: 240
# CHECK-NEXT:   Augmentation data:     1A

.text
.global _start
_start:
 nop

.section .eh_frame, "a"
  .long 16   # Size
  .long 0x00 # ID
  .byte 0x01 # Version

  .byte 0x7A # Augmentation string: "zR"
  .byte 0x52
  .byte 0x00

  .byte 0x01 # Code alignment factor, ULEB128
  .byte 0x01 # Data alignment factor, ULEB128
  
  .byte 0xF0 # Return address register, ubyte for version 1.

  .byte 0x01 # LEB128
  .byte 0x1A # DW_EH_PE_pcrel | DW_EH_PE_sdata2

  .byte 0x00
  .byte 0x00
  .byte 0x00

  .long 10   # Size
  .long 24   # ID
fde:
  .long _start - fde
  .word 0
