// RUN: split-file %s %t

// Test that .amdgcn_target emits separate warnings for conflicting xnack and
// sramecc settings between two .amdgcn_target directives. The first directive
// establishes the target id's xnack/sramecc modes; a second directive with
// different specific modes conflicts.

// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa %t/xnack.s 2>&1 | FileCheck --check-prefix=XNACK --implicit-check-not=warning %s
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa %t/sramecc.s 2>&1 | FileCheck --check-prefix=SRAMECC --implicit-check-not=warning %s
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa %t/both.s 2>&1 | FileCheck --check-prefix=BOTH --implicit-check-not=warning %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 %t/xnack-legacy.s 2>&1 | FileCheck --check-prefix=XNACK-LEGACY --implicit-check-not=warning %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 %t/sramecc-legacy.s 2>&1 | FileCheck --check-prefix=SRAMECC-LEGACY --implicit-check-not=warning %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 %t/both-legacy.s 2>&1 | FileCheck --check-prefix=BOTH-LEGACY --implicit-check-not=warning %s

// When a single directive specifies modes and nothing conflicts, no warning is
// emitted.
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa %t/noconflict.s 2>&1 | FileCheck --check-prefix=NOCONFLICT --implicit-check-not=warning %s

// The object emission path honors the directive's xnack/sramecc settings in the
// e_flags.
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa -filetype=obj %t/noconflict.s -o %t/both.o
// RUN: llvm-readobj --file-headers %t/both.o | FileCheck --check-prefix=OBJ %s

// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %t/noconflict-legacy.s -o %t/both-legacy.o
// RUN: llvm-readobj --file-headers %t/both-legacy.o | FileCheck --check-prefix=OBJ-LEGACY %s

//--- xnack.s
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack+"
// XNACK: warning: .amdgcn_target directive has conflicting xnack settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack-"

//--- sramecc.s
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack+"
// SRAMECC: warning: .amdgcn_target directive has conflicting sramecc settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack+"

//--- both.s
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack+"
// BOTH: warning: .amdgcn_target directive has conflicting xnack settings
// BOTH: warning: .amdgcn_target directive has conflicting sramecc settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"

//--- xnack-legacy.s
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack+"
// XNACK-LEGACY: warning: .amdgcn_target directive has conflicting xnack settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack-"

//--- sramecc-legacy.s
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack+"
// SRAMECC-LEGACY: warning: .amdgcn_target directive has conflicting sramecc settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack+"

//--- both-legacy.s
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc+:xnack+"
// BOTH-LEGACY: warning: .amdgcn_target directive has conflicting xnack settings
// BOTH-LEGACY: warning: .amdgcn_target directive has conflicting sramecc settings
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"

//--- noconflict.s
// NOCONFLICT: .amdgcn_target "amdgpu9.08-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"
// OBJ: EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 (0x800)
// OBJ: EF_AMDGPU_FEATURE_XNACK_OFF_V4 (0x200)
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"

//--- noconflict-legacy.s
// NOCONFLICT-LEGACY: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"
// OBJ-LEGACY: EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 (0x800)
// OBJ-LEGACY: EF_AMDGPU_FEATURE_XNACK_OFF_V4 (0x200)
.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx908:sramecc-:xnack-"
