# RUN: llvm-mc --verify -triple riscv32 -mattr=-relax -filetype obj -o /dev/null < %s
# RUN: llvm-mc --verify -triple riscv32 -mattr=+relax -filetype obj -o /dev/null < %s

1:
  addi a0, a0, %pcrel_lo(1b) # expected-error:17 {{could not find corresponding %pcrel_hi}}
  addi a0, a0, %pcrel_lo(0x123456) # expected-error:17 {{could not find corresponding %pcrel_hi}}
  addi a0, a0, %pcrel_lo(foo) # expected-error:17 {{could not find corresponding %pcrel_hi}}
