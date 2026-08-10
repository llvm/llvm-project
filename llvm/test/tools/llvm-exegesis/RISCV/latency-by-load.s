# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -opcode-name=LD 2>&1 | FileCheck --check-prefix=TEST1 %s

TEST1:      ---
TEST1-NEXT: mode: latency
TEST1-NEXT: key:
TEST1-NEXT:   instructions:
TEST1-NEXT:     - 'LD X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -opcode-name=LW 2>&1 | FileCheck --check-prefix=TEST2 %s

TEST2:      ---
TEST2-NEXT: mode: latency
TEST2-NEXT: key:
TEST2-NEXT:   instructions:
TEST2-NEXT:     - 'LW X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -opcode-name=LH 2>&1 | FileCheck --check-prefix=TEST3 %s

TEST3:      ---
TEST3-NEXT: mode: latency
TEST3-NEXT: key:
TEST3-NEXT:   instructions:
TEST3-NEXT:     - 'LH X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -opcode-name=LWU 2>&1 | FileCheck --check-prefix=TEST4 %s

TEST4:      ---
TEST4-NEXT: mode: latency
TEST4-NEXT: key:
TEST4-NEXT:   instructions:
TEST4-NEXT:     - 'LWU X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -opcode-name=LBU 2>&1 | FileCheck --check-prefix=TEST5 %s

TEST5:      ---
TEST5-NEXT: mode: latency
TEST5-NEXT: key:
TEST5-NEXT:   instructions:
TEST5-NEXT:     - 'LBU X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -opcode-name=LUI 2>&1 | FileCheck --check-prefix=TEST6 %s

TEST6: LUI: No strategy found to make the execution serial


# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -opcode-name=LB 2>&1 | FileCheck --check-prefix=TEST7 %s

TEST7:      ---
TEST7-NEXT: mode: latency
TEST7-NEXT: key:
TEST7-NEXT:   instructions:
TEST7-NEXT:     - 'LB X10 X10 i_0x0'

# RUN: llvm-exegesis -mode=latency --benchmark-phase=assemble-measured-code -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -mattr=+a -opcode-name=LR_W_RL 2>&1 | FileCheck --check-prefix=TEST8 %s

TEST8:      ---
TEST8-NEXT: mode: latency
TEST8-NEXT: key:
TEST8-NEXT:   instructions:
TEST8-NEXT:     - 'LR_W_RL X10 X10'
