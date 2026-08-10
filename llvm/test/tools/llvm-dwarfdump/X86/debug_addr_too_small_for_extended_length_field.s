# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: not llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: parsing address table at offset 0x0: unexpected end of data at offset 0xb while reading [0x4, 0xc)
# ERR-NOT: {{.}}

# too small section to contain an extended length field of a DWARF64 address table.
  .section  .debug_addr,"",@progbits
  .long 0xffffffff      # DWARF64 mark
  .space 7              # a DWARF64 unit length field takes 8 bytes.
