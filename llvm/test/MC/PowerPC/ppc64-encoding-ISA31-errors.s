# RUN: not llvm-mc -triple powerpc64-unknown-unknown < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple powerpc64le-unknown-unknown < %s 2> %t
# RUN: FileCheck < %t %s

 # CHECK: error: invalid operand for instruction
paddi 1, 1, 32, 1

# CHECK: error: invalid operand for instruction
pld 1, 32(1), 1

# CHECK: error: invalid operand for instruction
paddi 1, 1, 32, 1

# CHECK: error: invalid operand for instruction
plbz 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plfd 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plfs 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plha 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plhz 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plwa 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plwz 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plxsd 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plxssp 1, 32(1), 1

# CHECK: error: invalid operand for instruction
plxv 1, 32(1), 1

# CHECK: error: invalid operand for instruction
pstb 1, 32(1), 1

# CHECK: error: invalid operand for instruction
pstd 1, 32(1), 1

# CHECK: error: invalid operand for instruction
pstfd 1, 32(1), 1

# CHECK: error: invalid operand for instruction
pstfs 1, 32(1), 1

# CHECK: error: invalid operand for instruction
psth 1, 32(1), 1

# CHECK: error: invalid operand for instruction
pstw 1, 32(1), 1

# CHECK: error: invalid operand for instruction
pstxsd 1, 32(1), 1

# CHECK: error: invalid operand for instruction
pstxssp 1, 32(1), 1

# CHECK: error: invalid operand for instruction
pstxv 1, 32(1), 1

