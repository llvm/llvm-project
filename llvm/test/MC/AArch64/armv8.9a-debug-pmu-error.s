// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding               -mattr=+ite < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.8a -mattr=+ite < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.9a -mattr=+ite < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.3a -mattr=+ite < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.4a -mattr=+ite < %s 2>&1 | FileCheck %s

// FEAT_PMUv3p9/FEAT_PMUV3_ICNTR - PMZR_EL0 is write-only
            mrs x3, PMZR_EL0
// CHECK: [[@LINE-1]]:21: error: expected readable system register
