# RUN: not llvm-mc -triple=riscv32 -mattr=+svinval < %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=NO-H
# RUN: not llvm-mc -triple=riscv64 -mattr=+svinval < %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=NO-H
# RUN: not llvm-mc -triple=riscv32 -mattr=+h < %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=NO-SVINVAL
# RUN: not llvm-mc -triple=riscv64 -mattr=+h < %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=NO-SVINVAL

hinval.vvma a0, a1
# NO-H: :[[@LINE-1]]:1: error: instruction requires the following: 'H' (Hypervisor)
# NO-SVINVAL: :[[@LINE-2]]:1: error: instruction requires the following: 'Svinval' (Fine-Grained Address-Translation Cache Invalidation)

hinval.gvma a0, a1
# NO-H: :[[@LINE-1]]:1: error: instruction requires the following: 'H' (Hypervisor)
# NO-SVINVAL: :[[@LINE-2]]:1: error: instruction requires the following: 'Svinval' (Fine-Grained Address-Translation Cache Invalidation)
