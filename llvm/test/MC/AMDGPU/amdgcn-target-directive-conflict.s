// RUN: split-file %s %t

// Test that .amdgcn_target emits separate warnings for conflicting xnack and
// sramecc settings between the directive and the command line.

// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack,+sramecc %t/xnack.s 2>&1 | FileCheck --check-prefix=XNACK --implicit-check-not=warning %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack,+sramecc %t/sramecc.s 2>&1 | FileCheck --check-prefix=SRAMECC --implicit-check-not=warning %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack,+sramecc %t/both.s 2>&1 | FileCheck --check-prefix=BOTH --implicit-check-not=warning %s

// When the directive specifies modes but the command line leaves them
// unspecified (Any), there is no conflict and no warning is emitted.
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx908 %t/both.s 2>&1 | FileCheck --check-prefix=NOCONFLICT --implicit-check-not=warning %s

// The object emission path honors the directive's xnack/sramecc settings in the
// e_flags even when the command line does not specify them.
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %t/both.s -o %t/both.o
// RUN: llvm-readobj --file-headers %t/both.o | FileCheck --check-prefix=OBJ %s

//--- xnack.s
// XNACK: warning: .amdgcn_target directive has conflicting xnack settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack-"

//--- sramecc.s
// SRAMECC: warning: .amdgcn_target directive has conflicting sramecc settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack+"

//--- both.s
// BOTH: warning: .amdgcn_target directive has conflicting xnack settings
// BOTH: warning: .amdgcn_target directive has conflicting sramecc settings
// NOCONFLICT: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"
// OBJ: EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 (0x800)
// OBJ: EF_AMDGPU_FEATURE_XNACK_OFF_V4 (0x200)
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"
