# RUN: llvm-mc --triple=loongarch64 %s | FileCheck --check-prefix=PRINT %s
# RUN: not llvm-mc --filetype=obj --triple=loongarch64 %s -o /dev/null 2>&1 \
# RUN:			| FileCheck %s

# PRINT: .reloc 0, R_INVALID, 0
# CHECK: {{.*}}.s:[[# @LINE+1]]:11: error: unknown relocation name
.reloc 0, R_INVALID, 0
