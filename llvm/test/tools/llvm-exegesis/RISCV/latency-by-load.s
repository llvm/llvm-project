# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=LD | FileCheck --check-prefix=LD %s

LD:      ---
LD-NEXT: mode: latency
LD-NEXT: key:
LD-NEXT:   instructions:
LD-NEXT:     - 'LD X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=LW | FileCheck --check-prefix=LW %s

LW:      ---
LW-NEXT: mode: latency
LW-NEXT: key:
LW-NEXT:   instructions:
LW-NEXT:     - 'LW X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=LH | FileCheck --check-prefix=LH %s

LH:      ---
LH-NEXT: mode: latency
LH-NEXT: key:
LH-NEXT:   instructions:
LH-NEXT:     - 'LH X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=LWU | FileCheck --check-prefix=LWU %s

LWU:      ---
LWU-NEXT: mode: latency
LWU-NEXT: key:
LWU-NEXT:   instructions:
LWU-NEXT:     - 'LWU X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=LBU | FileCheck --check-prefix=LBU %s

LBU:      ---
LBU-NEXT: mode: latency
LBU-NEXT: key:
LBU-NEXT:   instructions:
LBU-NEXT:     - 'LBU X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=LUI 2>&1 | FileCheck --check-prefix=LUI %s

LUI: LUI: No strategy found to make the execution serial

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=LB | FileCheck --check-prefix=LB %s

LB:      ---
LB-NEXT: mode: latency
LB-NEXT: key:
LB-NEXT:   instructions:
LB-NEXT:     - 'LB X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=LR_W_RL -mattr="+a" | FileCheck --check-prefix=LR_W_RL %s

LR_W_RL:      ---
LR_W_RL-NEXT: mode: latency
LR_W_RL-NEXT: key:
LR_W_RL-NEXT:   instructions:
LR_W_RL-NEXT:     - 'LR_W_RL X10 X10'
