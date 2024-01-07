# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=+relax %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefixes=CHECK,RELAX %s
# RUN: llvm-mc --filetype=obj --triple=riscv64 --mattr=-relax %s \
# RUN:     | llvm-readobj -r - | FileCheck %s

.section ".dummy", "a"
.L1:
  call func
.p2align 3
.L2:
.dword .L2 - .L1
.word .L2 - .L1
.half .L2 - .L1
.byte .L2 - .L1

# CHECK:       Relocations [
# CHECK-NEXT:    Section ({{.*}}) .rela.dummy {
# CHECK-NEXT:      0x0 R_RISCV_CALL_PLT func 0x0
# RELAX-NEXT:      0x0 R_RISCV_RELAX - 0x0
# CHECK-NEXT:      0x8 R_RISCV_ADD64 .L2 0x0
# CHECK-NEXT:      0x8 R_RISCV_SUB64 .L1 0x0
# CHECK-NEXT:      0x10 R_RISCV_ADD32 .L2 0x0
# CHECK-NEXT:      0x10 R_RISCV_SUB32 .L1 0x0
# CHECK-NEXT:      0x14 R_RISCV_ADD16 .L2 0x0
# CHECK-NEXT:      0x14 R_RISCV_SUB16 .L1 0x0
# CHECK-NEXT:      0x16 R_RISCV_ADD8 .L2 0x0
# CHECK-NEXT:      0x16 R_RISCV_SUB8 .L1 0x0
# CHECK-NEXT:    }
# CHECK-NEXT:  ]
