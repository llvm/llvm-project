# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: section is not large enough to contain an extended length field of the address table at offset 0x0
# ERR-NOT: {{.}}

# too small section to contain an extended length field of a DWARF64 address table.
  .section  .debug_addr,"",@progbits
  .long 0xffffffff      # DWARF64 mark
  .space 7              # a DWARF64 unit length field takes 8 bytes.
