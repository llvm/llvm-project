# RUN: llvm-mc -triple=riscv64 %s 2>&1 | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc --verify -filetype=obj -triple=riscv64 %s -o /dev/null

# PRINT: .reloc {{.*}}, R_INVALID, 0
.reloc ., R_INVALID, 0 # expected-error:11 {{unknown relocation name}}
