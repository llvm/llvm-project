# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: .debug_addr table at offset 0x0 contains data of size 7 which is not a multiple of addr size 4
# ERR-NOT: {{.}}

# data size is not multiple of address_size
  .section  .debug_addr,"",@progbits
.Ldebug_addr0:
  .long 11 # unit_length = .short + .byte + .byte + .long + .long - 1
  .short 5 # version
  .byte 4  # address_size
  .byte 0  # segment_selector_size
  .long 0x00000000
  .long 0x00000001
