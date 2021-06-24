# RUN: llvm-mc -mattr -relax -triple riscv64 -filetype obj %s -o - | llvm-readobj -d -r - | FileCheck %s

.global function

# CHECK: .rela.text {

# Unrelaxed reference, this would normally fail, but the subsequent scoped
# relaxation forces relaxation on the file.
.dword function - .

# CHECK: 0x0 R_RISCV_ADD64 function 0x0
# CHECK-NEXT: 0x0 R_RISCV_SUB64 - 0x0

# Relaxed reference, this will resolve to a pair of `RISCV_ADD64` and
# `RISCV_SUB64` relocation.
.option push
.option relax
.dword function - .
.option pop

# CHECK: 0x8 R_RISCV_ADD64 function 0x0
# CHECK-NEXT: 0x8 R_RISCV_SUB64 - 0x0

# Unrelaxed reference, this will resolve to a pair of `RISCV_ADD64` and
# `RISCV_SUB64` relocation due to relaxation being sticky to the file.
.option push
.option norelax
.dword function - .
.option pop

# CHECK: 0x10 R_RISCV_ADD64 function 0x0
# CHECK-NEXT: 0x10 R_RISCV_SUB64 - 0x0

# CHECK: }
