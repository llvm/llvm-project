# RUN: llvm-mc --verify -triple riscv32 < %s
# RUN: llvm-mc --verify -triple riscv64 < %s

.cfi_startproc
.cfi_offset x00, 0 # expected-error:16 {{invalid register name}}
.cfi_offset a8, 8 # expected-error:15 {{invalid register name}}
.cfi_endproc
