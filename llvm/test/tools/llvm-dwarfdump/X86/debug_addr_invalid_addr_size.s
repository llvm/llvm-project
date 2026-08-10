# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: not llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: unsupported address size: 3 (supported are 2, 4, 8)
# ERR-NOT: {{.}}

# invalid addr size
  .section  .debug_addr,"",@progbits
.Ldebug_addr0:
  .long 12 # unit_length = .short + .byte + .byte + .long + .long
  .short 5 # version
  .byte 3  # address_size
  .byte 0  # segment_selector_size
  .long 0x00000000
  .long 0x00000001
