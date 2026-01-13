# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=AMOAND_D -mattr="+a" | FileCheck --check-prefix=AMOAND_D %s

AMOAND_D:      ---
AMOAND_D-NEXT: mode: latency
AMOAND_D-NEXT: key:
AMOAND_D-NEXT:   instructions:
AMOAND_D-NEXT:     - 'AMOAND_D [[RE01:X[0-9]+]] [[RE01:X[0-9]+]] X10'
AMOAND_D-NEXT: config: ''
AMOAND_D-NEXT: register_initial_values:
AMOAND_D-NEXT: - '[[RE01:X[0-9]+]]=0x0'
AMOAND_D-DAG: ...

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=AMOADD_W -mattr="+a" | FileCheck --check-prefix=AMOADD_W %s

AMOADD_W:      ---
AMOADD_W-NEXT: mode: latency
AMOADD_W-NEXT: key:
AMOADD_W-NEXT:   instructions:
AMOADD_W-NEXT:     - 'AMOADD_W [[RE02:X[0-9]+]] [[RE02:X[0-9]+]] X10'
AMOADD_W-NEXT: config: ''
AMOADD_W-NEXT: register_initial_values:
AMOADD_W-NEXT: - '[[RE02:X[0-9]+]]=0x0'
AMOADD_W-DAG: ...

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=AMOMAXU_D -mattr="+a" | FileCheck --check-prefix=AMOMAXU_D %s

AMOMAXU_D:      ---
AMOMAXU_D-NEXT: mode: latency
AMOMAXU_D-NEXT: key:
AMOMAXU_D-NEXT:   instructions:
AMOMAXU_D-NEXT:     - 'AMOMAXU_D [[RE03:X[0-9]+]] [[RE03:X[0-9]+]] X10'
AMOMAXU_D-NEXT: config: ''
AMOMAXU_D-NEXT: register_initial_values:
AMOMAXU_D-NEXT: - '[[RE03:X[0-9]+]]=0x0'
AMOMAXU_D-DAG: ...

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=AMOMIN_W -mattr="+a" | FileCheck --check-prefix=AMOMIN_W %s

AMOMIN_W:      ---
AMOMIN_W-NEXT: mode: latency
AMOMIN_W-NEXT: key:
AMOMIN_W-NEXT:   instructions:
AMOMIN_W-NEXT:     - 'AMOMIN_W [[RE04:X[0-9]+]] [[RE04:X[0-9]+]] X10'
AMOMIN_W-NEXT: config: ''
AMOMIN_W-NEXT: register_initial_values:
AMOMIN_W-NEXT: - '[[RE04:X[0-9]+]]=0x0'
AMOMIN_W-DAG: ...

# RUN: llvm-exegesis -mode=latency -mtriple=riscv64-unknown-linux-gnu --mcpu=generic --benchmark-phase=assemble-measured-code -opcode-name=AMOXOR_D -mattr="+a" | FileCheck --check-prefix=AMOXOR_D %s

AMOXOR_D:      ---
AMOXOR_D-NEXT: mode: latency
AMOXOR_D-NEXT: key:
AMOXOR_D-NEXT:   instructions:
AMOXOR_D-NEXT:     - 'AMOXOR_D [[RE05:X[0-9]+]] [[RE05:X[0-9]+]] X10'
AMOXOR_D-NEXT: config: ''
AMOXOR_D-NEXT: register_initial_values:
AMOXOR_D-NEXT: - '[[RE05:X[0-9]+]]=0x0'
AMOXOR_D-DAG: ...
