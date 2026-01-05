# RUN: not llvm-mc -triple xtensa --mattr=+debug,+density %s 2>&1 | FileCheck %s

LBL0:

# Out of range immediates

# uimm4
break 16, 0
# CHECK: :[[#@LINE-1]]:7: error: expected immediate in range [0, 15]
