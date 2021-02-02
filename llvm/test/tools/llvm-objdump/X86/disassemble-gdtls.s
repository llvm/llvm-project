# RUN: llvm-mc %s -filetype=obj -triple=x86_64 | llvm-objdump -d - | FileCheck %s

# CHECK:      <PR48901>:
# TODO: Should display data16 prefixes.
# CHECK-NEXT: 0: 66 48 8d 3d 00 00 00 00       leaq    (%rip), %rdi  # 8 <PR48901+0x8>
# CHECK-NEXT: 8: 66 66 48 e8 00 00 00 00       callq   0x10 <PR48901+0x10>
# CHECK-EMPTY:

PR48901:
 data16
 leaq   bar@TLSGD(%rip),%rdi
 data16
 data16
 rex64
 callq  __tls_get_addr@PLT

.section .tdata,"awT",@progbits
bar:
.long 42
