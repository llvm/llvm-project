## Check the data diff (separated by aligment directives) directives which in
## a section which has instructions but is not executable should generate relocs
## because it can not be calculated out in AttemptToFoldSymbolOffsetDifference.
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
# CHECK-NEXT:      0x8 R_RISCV_ADD64 .L2 0x0
# CHECK-NEXT:      0x8 R_RISCV_SUB64 .L1 0x0
# CHECK-NEXT:    }
# CHECK-NEXT:  ]
