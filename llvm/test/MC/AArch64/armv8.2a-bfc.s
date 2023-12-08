// RUN: llvm-mc -triple aarch64-linux-gnu -o - %s | FileCheck %s --check-prefix=BFI
// RUN: llvm-mc -triple aarch64-linux-gnu -o - %s -mattr=+v8.1a | FileCheck %s --check-prefix=BFI
// RUN: llvm-mc -triple aarch64-linux-gnu -o - %s -mattr=-v8.2a | FileCheck %s --check-prefix=BFI
// RUN: llvm-mc -triple aarch64-linux-gnu -o - %s -mattr=+v8.2a | FileCheck %s --check-prefix=BFC
// RUN: llvm-mc -triple aarch64-linux-gnu -o - %s -mattr=+v8.3a | FileCheck %s --check-prefix=BFC
// RUN: llvm-mc -triple aarch64-linux-gnu -o - %s -mattr=+v8.4a | FileCheck %s --check-prefix=BFC
// RUN: llvm-mc -triple aarch64-linux-gnu -o - %s -mattr=+v8.5a | FileCheck %s --check-prefix=BFC
  bfc w0, #1, #5

// BFI: bfi w0, wzr, #1, #5
// BFC: bfc w0, #1, #5
