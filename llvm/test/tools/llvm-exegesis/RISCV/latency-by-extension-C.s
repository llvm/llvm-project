# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=C_ADDI -mattr=+c | FileCheck --check-prefix=TEST1 %s

TEST1:      ---
TEST1-NEXT: mode: latency
TEST1-NEXT: key:
TEST1-NEXT:   instructions:
TEST1-NEXT:     - 'C_ADDI [[REG01:X[0-9]+]] [[RE02:X[0-9]+]] [[IMM0:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=C_ADDIW -mattr=+c | FileCheck --check-prefix=TEST2 %s

TEST2:      ---
TEST2-NEXT: mode: latency
TEST2-NEXT: key:
TEST2-NEXT:   instructions:
TEST2-NEXT:     - 'C_ADDIW [[REG11:X[0-9]+]] [[RE12:X[0-9]+]] [[IMM1:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=C_ANDI -mattr=+c | FileCheck --check-prefix=TEST3 %s

TEST3:      ---
TEST3-NEXT: mode: latency
TEST3-NEXT: key:
TEST3-NEXT:   instructions:
TEST3-NEXT:     - 'C_ANDI [[REG31:X[0-9]+]] [[REG32:X[0-9]+]] [[IMM3:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=C_SLLI -mattr=+c | FileCheck --check-prefix=TEST4 %s

TEST4:      ---
TEST4-NEXT: mode: latency
TEST4-NEXT: key:
TEST4-NEXT:   instructions:
TEST4-NEXT:     - 'C_SLLI [[REG81:X[0-9]+]] [[REG82:X[0-9]+]] [[IMM8:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=C_SRAI -mattr=+c | FileCheck --check-prefix=TEST5 %s

TEST5:      ---
TEST5-NEXT: mode: latency
TEST5-NEXT: key:
TEST5-NEXT:   instructions:
TEST5-NEXT:     - 'C_SRAI [[REG91:X[0-9]+]] [[REG92:X[0-9]+]] [[IMM9:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=C_SRLI -mattr=+c | FileCheck --check-prefix=TEST6 %s

TEST6:      ---
TEST6-NEXT: mode: latency
TEST6-NEXT: key:
TEST6-NEXT:   instructions:
TEST6-NEXT:     - 'C_SRLI [[REG101:X[0-9]+]] [[REG102:X[0-9]+]] [[IMM10:i_0x[0-9]+]]'
TEST6-LAST: ...


# RUN: llvm-exegesis  -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=C_LD -mattr=+c | FileCheck --check-prefix=TEST7 %s

TEST7:      ---
TEST7-NEXT: mode: latency
TEST7-NEXT: key:
TEST7-NEXT:   instructions:
TEST7-NEXT:     - 'C_LD [[REG61:X[0-9]+]] [[REG62:X[0-9]+]] [[IMM6:i_0x[0-9]+]]'

# RUN: llvm-exegesis  -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=C_LW -mattr=+c | FileCheck --check-prefix=TEST8 %s

TEST8:      ---
TEST8-NEXT: mode: latency
TEST8-NEXT: key:
TEST8-NEXT:   instructions:
TEST8-NEXT:     - 'C_LW [[REG71:X[0-9]+]] [[REG72:X[0-9]+]] [[IMM7:i_0x[0-9]+]]'
