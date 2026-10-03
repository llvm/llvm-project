# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_LWSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_LWSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_LWSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_LDSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_SWSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-STORE-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=C_SDSP --min-instructions=100000 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --repetition-mode=loop --loop-body-size=100 \
# RUN:   -mattr=+c --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-STORE-ASM --implicit-check-not='x2, x10'

# The floating-point and Zfinx stack-pointer-relative forms are pinned to X2
# by the very same encoding rule, so they must be handled without being
# enumerated one by one.

# RUN: llvm-exegesis --benchmark-phase=assemble-measured-code -opcode-name=C_FLWSP --min-instructions=100 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop \
# RUN:   --loop-body-size=100 -mattr=+c,+f --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-FLOAD-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis --benchmark-phase=assemble-measured-code -opcode-name=C_FLDSP --min-instructions=100 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --repetition-mode=loop \
# RUN:   --loop-body-size=100 -mattr=+c,+d --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-FLOAD-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis --benchmark-phase=assemble-measured-code -opcode-name=C_FSWSP --min-instructions=100 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop \
# RUN:   --loop-body-size=100 -mattr=+c,+f --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-FSTORE-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis --benchmark-phase=assemble-measured-code -opcode-name=C_FSDSP --min-instructions=100 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --repetition-mode=loop \
# RUN:   --loop-body-size=100 -mattr=+c,+d --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-FSTORE-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis --benchmark-phase=assemble-measured-code -opcode-name=C_LWSP_INX --min-instructions=100 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop \
# RUN:   --loop-body-size=100 -mattr=+c,+zfinx --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-LOAD-ASM --implicit-check-not='x2, x10'

# RUN: llvm-exegesis --benchmark-phase=assemble-measured-code -opcode-name=C_SWSP_INX --min-instructions=100 \
# RUN:   --dump-object-to-disk=%t.o -mtriple=riscv32-unknown-linux-gnu --mcpu=generic --repetition-mode=loop \
# RUN:   --loop-body-size=100 -mattr=+c,+zfinx --mode=inverse_throughput
# RUN: llvm-objdump -M numeric -d %t.o > %t.s
# RUN: FileCheck %s < %t.s --check-prefix=CHECK-STORE-ASM --implicit-check-not='x2, x10'

# These opcodes encode X2 as their base register. The scratch space they
# address is reserved in the function frame, so the stack pointer keeps its
# value and no copy of the scratch memory pointer into x2 is emitted -- the
# --implicit-check-not above rejects one if it ever comes back.

CHECK-LOAD-ASM: addi x2, x2, -0x{{[0-9a-f]+}}
CHECK-LOAD-ASM-COUNT-100: l{{[wd]}} x{{[0-9]+}}, 0x0(x2)

CHECK-STORE-ASM: addi x2, x2, -0x{{[0-9a-f]+}}
CHECK-STORE-ASM-COUNT-100: s{{[wd]}} x{{[0-9]+}}, 0x0(x2)

CHECK-FLOAD-ASM: addi x2, x2, -0x{{[0-9a-f]+}}
CHECK-FLOAD-ASM-COUNT-100: fl{{[wd]}} f{{[0-9]+}}, 0x0(x2)

CHECK-FSTORE-ASM: addi x2, x2, -0x{{[0-9a-f]+}}
CHECK-FSTORE-ASM-COUNT-100: fs{{[wd]}} f{{[0-9]+}}, 0x0(x2)
