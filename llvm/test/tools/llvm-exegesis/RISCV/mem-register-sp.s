# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_LWSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%d -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c
# RUN: llvm-objdump -M numeric -d %d > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_LWSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%d -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %d > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_LWSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%d -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c
# RUN: llvm-objdump -M numeric -d %d > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_LDSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%d -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c
# RUN: llvm-objdump -M numeric -d %d > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_SWSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%d -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %d > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-STORE-ASM

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_SDSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%d -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %d > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-STORE-ASM

CHECK-LOAD-ASM: mv x2, x10
CHECK-LOAD-ASM-COUNT-100: l{{[wd]}} x{{[0-9]+}}, 0x0(x2)

CHECK-STORE-ASM: mv x2, x10
CHECK-STORE-ASM-COUNT-100: s{{[wd]}} x{{[0-9]+}}, 0x0(x2)
