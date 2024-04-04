# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=AMOAND_D -mattr="+a" | FileCheck --check-prefix=TEST1 %s

TEST1:      ---
TEST1-NEXT: mode: latency
TEST1-NEXT: key:
TEST1-NEXT:   instructions:
TEST1-NEXT:     - 'AMOAND_D [[RE01:X[0-9]+]] X10 [[RE01:X[0-9]+]]'
TEST1-NEXT: config: ''
TEST1-NEXT: register_initial_values:
TEST1-NEXT: - '[[RE01:X[0-9]+]]=0x0'
TEST1-LAST: ...

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=AMOADD_W -mattr="+a" | FileCheck --check-prefix=TEST2 %s

TEST2:      ---
TEST2-NEXT: mode: latency
TEST2-NEXT: key:
TEST2-NEXT:   instructions:
TEST2-NEXT:     - 'AMOADD_W [[RE02:X[0-9]+]] X10 [[RE02:X[0-9]+]]'
TEST2-NEXT: config: ''
TEST2-NEXT: register_initial_values:
TEST2-NEXT: - '[[RE02:X[0-9]+]]=0x0'
TEST2-LAST: ...

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=AMOMAXU_D -mattr="+a" | FileCheck --check-prefix=TEST3 %s

TEST3:      ---
TEST3-NEXT: mode: latency
TEST3-NEXT: key:
TEST3-NEXT:   instructions:
TEST3-NEXT:     - 'AMOMAXU_D [[RE03:X[0-9]+]] X10 [[RE03:X[0-9]+]]'
TEST3-NEXT: config: ''
TEST3-NEXT: register_initial_values:
TEST3-NEXT: - '[[RE03:X[0-9]+]]=0x0'
TEST3-LAST: ...

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=AMOMIN_W -mattr="+a" | FileCheck --check-prefix=TEST4 %s

TEST4:      ---
TEST4-NEXT: mode: latency
TEST4-NEXT: key:
TEST4-NEXT:   instructions:
TEST4-NEXT:     - 'AMOMIN_W [[RE04:X[0-9]+]] X10 [[RE04:X[0-9]+]]'
TEST4-NEXT: config: ''
TEST4-NEXT: register_initial_values:
TEST4-NEXT: - '[[RE04:X[0-9]+]]=0x0'
TEST4-LAST: ...

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --benchmark-phase=assemble-measured-code -opcode-name=AMOXOR_D -mattr="+a" | FileCheck --check-prefix=TEST5 %s

TEST5:      ---
TEST5-NEXT: mode: latency
TEST5-NEXT: key:
TEST5-NEXT:   instructions:
TEST5-NEXT:     - 'AMOXOR_D [[RE05:X[0-9]+]] X10 [[RE05:X[0-9]+]]'
TEST5-NEXT: config: ''
TEST5-NEXT: register_initial_values:
TEST5-NEXT: - '[[RE05:X[0-9]+]]=0x0'
TEST5-LAST: ...
