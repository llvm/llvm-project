# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64r6 -target-abi n64 2>&1 -mattr=+micromips | FileCheck %s --check-prefix=64R6
# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64r6 -target-abi n32 2>&1 -mattr=+micromips | FileCheck %s --check-prefix=64R6

# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64 -target-abi n64 2>&1 -mattr=+micromips | FileCheck %s --check-prefix=64
# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64 -target-abi n32 2>&1 -mattr=+micromips | FileCheck %s --check-prefix=64

# 64R6: microMIPS64R6 is not supported
# 64:   microMIPS64 is not supported
