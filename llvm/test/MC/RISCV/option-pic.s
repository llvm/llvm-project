# RUN: llvm-mc -triple riscv32 -mattr=-relax -M no-aliases < %s \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

# RUN: llvm-mc -triple riscv32 -mattr=-relax -M no-aliases \
# RUN:     -position-independent < %s | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -position-independent < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

# RUN: llvm-mc -triple riscv64 -mattr=-relax -M no-aliases < %s \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

# RUN: llvm-mc -triple riscv64 -mattr=-relax -M no-aliases \
# RUN:     -position-independent < %s | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -position-independent < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

.option pic
# CHECK-INST: .option pic

la s0, symbol
# CHECK-INST: auipc	s0, %got_pcrel_hi(symbol)
# CHECK-INST: l{{[wd]}}	s0, %pcrel_lo(.Lpcrel_hi0)(s0)
# CHECK-RELOC: R_RISCV_GOT_HI20 symbol 0x0
# CHECK-RELOC: R_RISCV_PCREL_LO12_I .Lpcrel_hi0 0x0
