// RUN: split-file %s %t

// Test that .amdgcn_target emits separate warnings for conflicting xnack and
// sramecc settings between the directive and the command line.

// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa -mattr=+xnack,+sramecc %t/xnack.s 2>&1 | FileCheck --check-prefix=XNACK --implicit-check-not=warning %s
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa -mattr=+xnack,+sramecc %t/sramecc.s 2>&1 | FileCheck --check-prefix=SRAMECC --implicit-check-not=warning %s
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa -mattr=+xnack,+sramecc %t/both.s 2>&1 | FileCheck --check-prefix=BOTH --implicit-check-not=warning %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack,+sramecc %t/xnack-legacy.s 2>&1 | FileCheck --check-prefix=XNACK-LEGACY --implicit-check-not=warning %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack,+sramecc %t/sramecc-legacy.s 2>&1 | FileCheck --check-prefix=SRAMECC-LEGACY --implicit-check-not=warning %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack,+sramecc %t/both-legacy.s 2>&1 | FileCheck --check-prefix=BOTH-LEGACY --implicit-check-not=warning %s

// When the directive specifies modes but the command line leaves them
// unspecified (Any), there is no conflict and no warning is emitted.
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa %t/both.s 2>&1 | FileCheck --check-prefix=NOCONFLICT --implicit-check-not=warning %s

// The object emission path honors the directive's xnack/sramecc settings in the
// e_flags even when the command line does not specify them.
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa -filetype=obj %t/both.s -o %t/both.o
// RUN: llvm-readobj --file-headers %t/both.o | FileCheck --check-prefix=OBJ %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %t/both-legacy.s -o %t/both-legacy.o
// RUN: llvm-readobj --file-headers %t/both-legacy.o | FileCheck --check-prefix=OBJ-LEGACY %s

//--- xnack.s
// XNACK: warning: .amdgcn_target directive has conflicting xnack settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack-"

//--- sramecc.s
// SRAMECC: warning: .amdgcn_target directive has conflicting sramecc settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack+"

//--- both.s
// BOTH: warning: .amdgcn_target directive has conflicting xnack settings
// BOTH: warning: .amdgcn_target directive has conflicting sramecc settings
// NOCONFLICT: .amdgcn_target "amdgpu9.08-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"
// OBJ: EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 (0x800)
// OBJ: EF_AMDGPU_FEATURE_XNACK_OFF_V4 (0x200)
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"

//--- xnack-legacy.s
// XNACK-LEGACY: warning: .amdgcn_target directive has conflicting xnack settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack-"

//--- sramecc-legacy.s
// SRAMECC-LEGACY: warning: .amdgcn_target directive has conflicting sramecc settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack+"

//--- both-legacy.s
// BOTH-LEGACY: warning: .amdgcn_target directive has conflicting xnack settings
// BOTH-LEGACY: warning: .amdgcn_target directive has conflicting sramecc settings
// NOCONFLICT-LEGACY: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"
// OBJ-LEGACY: EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 (0x800)
// OBJ-LEGACY: EF_AMDGPU_FEATURE_XNACK_OFF_V4 (0x200)
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"
