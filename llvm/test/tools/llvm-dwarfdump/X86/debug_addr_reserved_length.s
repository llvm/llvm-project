# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o - | \
# RUN:   llvm-dwarfdump --debug-addr - 2>&1 | \
# RUN:   FileCheck %s --implicit-check-not=error

# CHECK: error: parsing address table at offset 0x0: unsupported reserved unit length of value 0xfffffff0

.section .debug_addr,"",@progbits
.long 0xfffffff0
