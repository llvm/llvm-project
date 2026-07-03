## A label difference separated by an alignment directive, when the
## referenced symbols are in a non-executable section with instructions,
## should generate ADD/SUB relocations.
## https://github.com/llvm/llvm-project/pull/76552

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

# CHECK:       Relocations [
# CHECK-NEXT:    Section ({{.*}}) .rela.dummy {
# CHECK-NEXT:      0x0 R_RISCV_CALL_PLT func 0x0
# RELAX-NEXT:      0x0 R_RISCV_RELAX - 0x0
# RELAX-NEXT:      0x8 R_RISCV_ADD64 .L2 0x0
# RELAX-NEXT:      0x8 R_RISCV_SUB64 .L1 0x0
# CHECK-NEXT:    }
# CHECK-NEXT:  ]
