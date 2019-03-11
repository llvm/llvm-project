# RUN: llvm-mc -triple=riscv32 -target-abi foo < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32I-FOO %s
# RUN: llvm-mc -triple=riscv32 -mattr=+f -target-abi ilp32foof < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32IF-ILP32FOOF %s

# RV32I-FOO: 'foo' is not a recognized ABI for this target (ignoring target-abi)
# RV32IF-ILP32FOOF: 'ilp32foof' is not a recognized ABI for this target (ignoring target-abi)

# RUN: llvm-mc -triple=riscv64 -target-abi ilp32 < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64I-ILP32 %s
# RUN: llvm-mc -triple=riscv64 -mattr=+f -target-abi ilp32f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64IF-ILP32F %s
# RUN: llvm-mc -triple=riscv64 -mattr=+d -target-abi ilp32d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64IFD-ILP32D %s
# RUN: llvm-mc -triple=riscv64 -target-abi ilp32e < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64I-ILP32E %s

# RV64I-ILP32: 32-bit ABIs are not supported for 64-bit targets (ignoring target-abi)
# RV64IF-ILP32F: 32-bit ABIs are not supported for 64-bit targets (ignoring target-abi)
# RV64IFD-ILP32D: 32-bit ABIs are not supported for 64-bit targets (ignoring target-abi)
# RV64I-ILP32E: 32-bit ABIs are not supported for 64-bit targets (ignoring target-abi)

# RUN: llvm-mc -triple=riscv32 -target-abi lp64 < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32I-LP64 %s
# RUN: llvm-mc -triple=riscv32 -mattr=+f -target-abi lp64f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32IF-LP64F %s
# RUN: llvm-mc -triple=riscv32 -mattr=+d -target-abi lp64d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32IFD-LP64D %s

# RV32I-LP64: 64-bit ABIs are not supported for 32-bit targets (ignoring target-abi)
# RV32IF-LP64F: 64-bit ABIs are not supported for 32-bit targets (ignoring target-abi)
# RV32IFD-LP64D: 64-bit ABIs are not supported for 32-bit targets (ignoring target-abi)

# RUN: llvm-mc -triple=riscv32 -target-abi ilp32f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32I-ILP32F %s
# RUN: llvm-mc -triple=riscv64 -target-abi lp64f < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64I-LP64F %s

# RV32I-ILP32F: Hard-float 'f' ABI can't be used for a target that doesn't support the F instruction set extension (ignoring target-abi)
# RV64I-LP64F: Hard-float 'f' ABI can't be used for a target that doesn't support the F instruction set extension (ignoring target-abi)

# RUN: llvm-mc -triple=riscv32 -target-abi ilp32d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32I-ILP32D %s
# RUN: llvm-mc -triple=riscv32 -mattr=+f -target-abi ilp32d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV32IF-ILP32D %s
# RUN: llvm-mc -triple=riscv64 -target-abi lp64d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64I-LP64D %s
# RUN: llvm-mc -triple=riscv64 -mattr=+f -target-abi lp64d < %s 2>&1 \
# RUN:   | FileCheck -check-prefix=RV64IF-LP64D %s

# RV32I-ILP32D: Hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension (ignoring target-abi)
# RV32IF-ILP32D: Hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension (ignoring target-abi)
# RV64I-LP64D: Hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension (ignoring target-abi)
# RV64IF-LP64D: Hard-float 'd' ABI can't be used for a target that doesn't support the D instruction set extension (ignoring target-abi)

nop
