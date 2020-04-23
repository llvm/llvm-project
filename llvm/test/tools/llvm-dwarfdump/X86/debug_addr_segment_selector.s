# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: not llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: address table at offset 0x0 has unsupported segment selector size 1
# ERR-NOT: {{.}}

# non-zero segment_selector_size
# TODO: make this valid
  .section  .debug_addr,"",@progbits
.Ldebug_addr0:
  .long 4  # unit_length = .short + .byte + .byte
  .short 5 # version
  .byte 4  # address_size
  .byte 1  # segment_selector_size
