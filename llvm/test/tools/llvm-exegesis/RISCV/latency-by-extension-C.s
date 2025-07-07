# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=C_ADDI -mattr=+c | FileCheck --check-prefix=C_ADDI %s

C_ADDI:      ---
C_ADDI-NEXT: mode: latency
C_ADDI-NEXT: key:
C_ADDI-NEXT:   instructions:
C_ADDI-NEXT:     - 'C_ADDI [[REG01:X[0-9]+]] [[RE02:X[0-9]+]] [[IMM0:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=C_ADDIW -mattr=+c | FileCheck --check-prefix=C_ADDIW %s

C_ADDIW:      ---
C_ADDIW-NEXT: mode: latency
C_ADDIW-NEXT: key:
C_ADDIW-NEXT:   instructions:
C_ADDIW-NEXT:     - 'C_ADDIW [[REG11:X[0-9]+]] [[RE12:X[0-9]+]] [[IMM1:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=C_ANDI -mattr=+c | FileCheck --check-prefix=C_ANDI %s

C_ANDI:      ---
C_ANDI-NEXT: mode: latency
C_ANDI-NEXT: key:
C_ANDI-NEXT:   instructions:
C_ANDI-NEXT:     - 'C_ANDI [[REG31:X[0-9]+]] [[REG32:X[0-9]+]] [[IMM3:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=C_SLLI -mattr=+c | FileCheck --check-prefix=C_SLLI %s

C_SLLI:      ---
C_SLLI-NEXT: mode: latency
C_SLLI-NEXT: key:
C_SLLI-NEXT:   instructions:
C_SLLI-NEXT:     - 'C_SLLI [[REG81:X[0-9]+]] [[REG82:X[0-9]+]] [[IMM8:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=C_SRAI -mattr=+c | FileCheck --check-prefix=C_SRAI %s

C_SRAI:      ---
C_SRAI-NEXT: mode: latency
C_SRAI-NEXT: key:
C_SRAI-NEXT:   instructions:
C_SRAI-NEXT:     - 'C_SRAI [[REG91:X[0-9]+]] [[REG92:X[0-9]+]] [[IMM9:i_0x[0-9]+]]'

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=C_SRLI -mattr=+c | FileCheck --check-prefix=C_SRLI %s

C_SRLI:      ---
C_SRLI-NEXT: mode: latency
C_SRLI-NEXT: key:
C_SRLI-NEXT:   instructions:
C_SRLI-NEXT:     - 'C_SRLI [[REG101:X[0-9]+]] [[REG102:X[0-9]+]] [[IMM10:i_0x[0-9]+]]'
C_SRLI-DAG: ...
