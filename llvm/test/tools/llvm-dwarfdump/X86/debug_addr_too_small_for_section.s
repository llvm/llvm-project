# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: section is not large enough to contain an address table of length 0x10 at offset 0x0
# ERR-NOT: {{.}}

# too small section to contain section of given length
  .section  .debug_addr,"",@progbits
.Ldebug_addr0:
  .long 12 # unit_length
  .short 5 # version
  .byte 4  # address_size
  .byte 0  # segment_selector_size
