# RUN: not llvm-mc -triple=mips -mcpu=mips1 < %s 2>&1 \
# RUN:   | FileCheck --check-prefix=MIPS32 %s
# RUN: llvm-mc -triple=mips -show-encoding -mcpu=mips64 < %s \
# RUN:   | FileCheck --check-prefix=MIPS64 %s

sle   $4, $5, 0x100000000
# MIPS32: :[[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
# MIPS64: ori  $4, $zero, 32768  # encoding: [0x34,0x04,0x80,0x00]
# MIPS64: dsll $4, $4, 17        # encoding: [0x00,0x04,0x24,0x78]
# MIPS64: slt  $4, $4, $5        # encoding: [0x00,0x85,0x20,0x2a]
# MIPS64: xori $4, $4, 1         # encoding: [0x38,0x84,0x00,0x01]
sleu  $4, $5, 0x100000000
# MIPS32: :[[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
# MIPS64: ori  $4, $zero, 32768  # encoding: [0x34,0x04,0x80,0x00]
# MIPS64: dsll $4, $4, 17        # encoding: [0x00,0x04,0x24,0x78]
# MIPS64: sltu $4, $4, $5        # encoding: [0x00,0x85,0x20,0x2b]
# MIPS64: xori $4, $4, 1         # encoding: [0x38,0x84,0x00,0x01]
sle   $4, 0x100000000
# MIPS32: :[[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
# MIPS64: ori  $1, $zero, 32768  # encoding: [0x34,0x01,0x80,0x00]
# MIPS64: dsll $1, $1, 17        # encoding: [0x00,0x01,0x0c,0x78]
# MIPS64: slt  $4, $1, $4        # encoding: [0x00,0x24,0x20,0x2a]
# MIPS64: xori  $4, $4, 1        # encoding: [0x38,0x84,0x00,0x01]
sleu  $4, 0x100000000
# MIPS32: :[[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
# MIPS64: ori  $1, $zero, 32768  # encoding: [0x34,0x01,0x80,0x00]
# MIPS64: dsll $1, $1, 17        # encoding: [0x00,0x01,0x0c,0x78]
# MIPS64: sltu $4, $1, $4        # encoding: [0x00,0x24,0x20,0x2b]
# MIPS64: xori  $4, $4, 1        # encoding: [0x38,0x84,0x00,0x01]
