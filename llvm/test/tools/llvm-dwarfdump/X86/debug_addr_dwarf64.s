# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: DWARF64 is not supported in .debug_addr at offset 0x0
# ERR-NOT: {{.}}

# DWARF64 table
  .section  .debug_addr,"",@progbits
.Ldebug_addr0:
  .long 0xffffffff # unit_length DWARF64 mark
  .quad 12         # unit_length
  .short 5         # version
  .byte 3          # address_size
  .byte 0          # segment_selector_size
  .long 0x00000000
  .long 0x00000001
