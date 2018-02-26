# RUN: not llvm-mc %s -triple=riscv64 -riscv-no-aliases 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

rdinstreth x29 # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
rdcycleh x27   # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
rdtimeh x28    # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
