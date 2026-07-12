// Test that llvm-objdump emits the .amdgcn_target directive based on e_flags.

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t-gfx900.o
// RUN: llvm-objdump --disassemble-all %t-gfx900.o | FileCheck --check-prefix=CHECK-GFX900 %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t-gfx908.o
// RUN: llvm-objdump --disassemble-all %t-gfx908.o | FileCheck --check-prefix=CHECK-GFX908 %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1010 -filetype=obj %s -o %t-gfx1010.o
// RUN: llvm-objdump --disassemble-all %t-gfx1010.o | FileCheck --check-prefix=CHECK-GFX1010 %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t-gfx1100.o
// RUN: llvm-objdump --disassemble-all %t-gfx1100.o | FileCheck --check-prefix=CHECK-GFX1100 %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %s -o %t-gfx1200.o
// RUN: llvm-objdump --disassemble-all %t-gfx1200.o | FileCheck --check-prefix=CHECK-GFX1200 %s

// Test xnack/sramecc combinations
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t-default.o
// RUN: llvm-objdump --disassemble-all %t-default.o | FileCheck --check-prefix=CHECK-DEFAULT %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack,+sramecc -filetype=obj %s -o %t-both-on.o
// RUN: llvm-objdump --disassemble-all %t-both-on.o | FileCheck --check-prefix=CHECK-BOTH-ON %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-xnack,-sramecc -filetype=obj %s -o %t-both-off.o
// RUN: llvm-objdump --disassemble-all %t-both-off.o | FileCheck --check-prefix=CHECK-BOTH-OFF %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack,-sramecc -filetype=obj %s -o %t-xnack-on.o
// RUN: llvm-objdump --disassemble-all %t-xnack-on.o | FileCheck --check-prefix=CHECK-XNACK-ON %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-xnack,+sramecc -filetype=obj %s -o %t-sramecc-on.o
// RUN: llvm-objdump --disassemble-all %t-sramecc-on.o | FileCheck --check-prefix=CHECK-SRAMECC-ON %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack -filetype=obj %s -o %t-xnack-only.o
// RUN: llvm-objdump --disassemble-all %t-xnack-only.o | FileCheck --check-prefix=CHECK-XNACK-ONLY %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-xnack -filetype=obj %s -o %t-xnack-off-only.o
// RUN: llvm-objdump --disassemble-all %t-xnack-off-only.o | FileCheck --check-prefix=CHECK-XNACK-OFF-ONLY %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+sramecc -filetype=obj %s -o %t-sramecc-only.o
// RUN: llvm-objdump --disassemble-all %t-sramecc-only.o | FileCheck --check-prefix=CHECK-SRAMECC-ONLY %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-sramecc -filetype=obj %s -o %t-sramecc-off-only.o
// RUN: llvm-objdump --disassemble-all %t-sramecc-off-only.o | FileCheck --check-prefix=CHECK-SRAMECC-OFF-ONLY %s

// CHECK-GFX900: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx900"

// CHECK-GFX908: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908"

// CHECK-GFX1010: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1010"

// CHECK-GFX1100: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1100"

// CHECK-GFX1200: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1200"

// CHECK-DEFAULT: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908"

// CHECK-BOTH-ON: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908:sramecc+:xnack+"

// CHECK-BOTH-OFF: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"

// CHECK-XNACK-ON: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908:sramecc-:xnack+"

// CHECK-SRAMECC-ON: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908:sramecc+:xnack-"

// CHECK-XNACK-ONLY: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908:xnack+"

// CHECK-XNACK-OFF-ONLY: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908:xnack-"

// CHECK-SRAMECC-ONLY: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908:sramecc+"

// CHECK-SRAMECC-OFF-ONLY: .amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx908:sramecc-"
