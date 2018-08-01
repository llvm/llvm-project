# RUN: llvm-mc %s -filetype obj -triple i386-pc-linux -o - | \
# RUN: llvm-dwarfdump -debug-addr - 2> %t.err | FileCheck %s
# RUN: FileCheck %s -input-file %t.err -check-prefix=ERR

# CHECK: .debug_addr contents:
# CHECK-NOT: {{.}}
# ERR: section is not large enough to contain a .debug_addr table length at offset 0x0
# ERR-NOT: {{.}}

# too small section to contain length field
  .section  .debug_addr,"",@progbits
.Ldebug_addr0:
  .short 1 # unit_length
