# RUN: not llvm-mc -triple=riscv32 -target-abi foo < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32I-FOO %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+f -target-abi ilp32foof < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32IF-ILP32FOOF %s

# RV32I-FOO: <stdin>:1:1: error: 'foo' is not a recognized ABI for this target
# RV32IF-ILP32FOOF: <stdin>:1:1: error: 'ilp32foof' is not a recognized ABI for this target

# RUN: not llvm-mc -triple=riscv64 -target-abi ilp32 < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64I-ILP32 %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+f -target-abi ilp32f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64IF-ILP32F %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+d -target-abi ilp32d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64IFD-ILP32D %s
# RUN: not llvm-mc -triple=riscv64 -target-abi ilp32e < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64I-ILP32E %s

# RV64I-ILP32: <stdin>:1:1: error: 32-bit ABIs are not supported for 64-bit targets
# RV64IF-ILP32F: <stdin>:1:1: error: 32-bit ABIs are not supported for 64-bit targets
# RV64IFD-ILP32D: <stdin>:1:1: error: 32-bit ABIs are not supported for 64-bit targets
# RV64I-ILP32E: <stdin>:1:1: error: 32-bit ABIs are not supported for 64-bit targets

# RUN: not llvm-mc -triple=riscv32 -target-abi lp64 < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32I-LP64 %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+f -target-abi lp64f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32IF-LP64F %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+d -target-abi lp64d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32IFD-LP64D %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+e -target-abi lp64 < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32E-LP64 %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+e,+f -target-abi lp64f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32EF-LP64F %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+e,+d -target-abi lp64d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32EFD-LP64D %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+e -target-abi lp64e < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32E-LP64E %s

# RV32I-LP64: <stdin>:1:1: error: 64-bit ABIs are not supported for 32-bit targets
# RV32IF-LP64F: <stdin>:1:1: error: 64-bit ABIs are not supported for 32-bit targets
# RV32IFD-LP64D: <stdin>:1:1: error: 64-bit ABIs are not supported for 32-bit targets
# RV32E-LP64: <stdin>:1:1: error: 64-bit ABIs are not supported for 32-bit targets
# RV32EF-LP64F: <stdin>:1:1: error: 64-bit ABIs are not supported for 32-bit targets
# RV32EFD-LP64D: <stdin>:1:1: error: 64-bit ABIs are not supported for 32-bit targets
# RV32E-LP64E: <stdin>:1:1: error: 64-bit ABIs are not supported for 32-bit targets

# An explicit ABI that matches the RVE requirement (so it isn't rejected by earlier checks)
# RUN: not llvm-mc -triple=riscv32 -mattr=+e,+d -target-abi ilp32e < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32E-ILP32E-D %s
# RV32E-ILP32E-D: LLVM ERROR: ILP32E cannot be used with the D ISA extension

# RUN: not llvm-mc -triple=riscv32 -target-abi ilp32f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32I-ILP32F %s
# RUN: not llvm-mc -triple=riscv64 -target-abi lp64f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64I-LP64F %s

# RV32I-ILP32F: <stdin>:1:1: error: hard-float 'f' ABI can't be used for a target that doesn't support the F instruction set extension
# RV64I-LP64F: <stdin>:1:1: error: hard-float 'f' ABI can't be used for a target that doesn't support the F instruction set extension

# RUN: not llvm-mc -triple=riscv32 -target-abi ilp32d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32I-ILP32D %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+f -target-abi ilp32d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32IF-ILP32D %s
# RUN: not llvm-mc -triple=riscv64 -target-abi lp64d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64I-LP64D %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+f -target-abi lp64d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64IF-LP64D %s

# RV32I-ILP32D: <stdin>:1:1: error: hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension
# RV32IF-ILP32D: <stdin>:1:1: error: hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension
# RV64I-LP64D: <stdin>:1:1: error: hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension
# RV64IF-LP64D: <stdin>:1:1: error: hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension

# RUN: not llvm-mc -triple=riscv32 -mattr=+e -target-abi ilp32 < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32EF-ILP32F %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+e,+f -target-abi ilp32f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32EF-ILP32F %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+e,+d -target-abi ilp32f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32EFD-ILP32F %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+e,+d -target-abi ilp32d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32EFD-ILP32D %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+e -target-abi cheriot < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32E-CHERIOT %s

# RV32E-ILP32: <stdin>:1:1: error: only the ilp32e ABI is supported for RV32E
# RV32EF-ILP32F: <stdin>:1:1: error: only the ilp32e ABI is supported for RV32E
# RV32EFD-ILP32F: <stdin>:1:1: error: only the ilp32e ABI is supported for RV32E
# RV32EFD-ILP32D: <stdin>:1:1: error: only the ilp32e ABI is supported for RV32E
# RV32E-CHERIOT: <stdin>:1:1: error: only the ilp32e ABI is supported for RV32E

# RUN: not llvm-mc -triple=riscv64 -mattr=+e -target-abi lp64 < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64EF-LP64F %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+e,+f -target-abi lp64f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64EF-LP64F %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+e,+d -target-abi lp64f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64EFD-LP64F %s
# RUN: not llvm-mc -triple=riscv64 -mattr=+e,+d -target-abi lp64d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64EFD-LP64D %s

# RV64E-LP64: <stdin>:1:1: error: only the lp64e ABI is supported for RV64E
# RV64EF-LP64F: <stdin>:1:1: error: only the lp64e ABI is supported for RV64E
# RV64EFD-LP64F: <stdin>:1:1: error: only the lp64e ABI is supported for RV64E
# RV64EFD-LP64D: <stdin>:1:1: error: only the lp64e ABI is supported for RV64E

# RUN: not llvm-mc -triple=riscv32 -mattr=+e,+xcheriot -target-abi ilp32e < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32EXCHERIOT-ILP32 %s

# RV32EXCHERIOT-ILP32: <stdin>:1:1: error: only the cheriot ABI is supported for XCheriot

nop
