# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-smctr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-smctr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-ssctr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-ssctr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-smctr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-smctr -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-smctr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-smctr -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-ssctr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-ssctr -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-ssctr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-ssctr -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# RUN: not llvm-mc -triple riscv32 -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -defsym=RV64=1 -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-INST: sctrclr
# CHECK: encoding: [0x73,0x00,0x40,0x10]
# CHECK-NO-EXT: error: instruction requires the following: 'Smctr' (Control Transfer Records Machine Level) or 'Ssctr' (Control Transfer Records Supervisor Level){{$}}
sctrclr
