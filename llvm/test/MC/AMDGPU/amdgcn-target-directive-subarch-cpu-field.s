// RUN: split-file %s %t
// RUN: llvm-mc -triple amdgpu12.50-amd-amdhsa %t/amdhsa.s | FileCheck --check-prefix=AMDHSA %s
// RUN: not llvm-mc -triple amdgpu12.5-amd-amdhsa --amdhsa-code-object-version=6 %t/amdhsa-generic-subarch-cpu.s -filetype=null 2>&1 | FileCheck --check-prefix=AMDHSA-GENERIC-SUBARCH %s
// RUN: llvm-mc -triple amdgpu12.50-amd-amdpal %t/amdpal.s | FileCheck --check-prefix=AMDPAL %s
// RUN: not llvm-mc -triple amdgpu12.5-amd-amdpal --amdhsa-code-object-version=6 %t/amdpal-generic-subarch-cpu.s -filetype=null 2>&1 | FileCheck --check-prefix=AMDPAL-GENERIC-SUBARCH %s
// RUN: llvm-mc -triple amdgpu12.50-amd-amdhsa %t/cpu.s | FileCheck --check-prefix=AMDHSA %s
// RUN: not llvm-mc -triple amdgpu12.50-amd-amdhsa %t/subarch-mismatch.s -filetype=null 2>&1 | FileCheck --check-prefix=ERR %s
// RUN: not llvm-mc -triple amdgpu12.50-amd-amdhsa %t/cpu-not-covered.s -filetype=null 2>&1 | FileCheck --check-prefix=NOTCOVERED %s
// RUN: not llvm-mc -triple amdgpu12.50-amd-amdhsa %t/cpu-not-covered-same-major.s -filetype=null 2>&1 | FileCheck --check-prefix=NOTCOVERED-SAME-MAJOR %s
// RUN: not llvm-mc -triple amdgpu12.50-amd-amdpal %t/isa-cpu-not-covered.s -filetype=null 2>&1 | FileCheck --check-prefix=ISA-NOTCOVERED %s

// Check how the CPU field is validated against the triple
// subarch. The processor name field may be empty, but if it's not it
// must be compatible with triple's subarch.

//--- amdhsa.s
// AMDHSA: .amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-"

//--- amdhsa-generic-subarch-cpu.s
// FIXME: This should be accepted and passed through as-is
// AMDHSA-GENERIC-SUBARCH: error: .amdgcn_target directive's target id amdgpu12.5-amd-amdhsa-unknown-gfx1251 does not match the specified target id amdgpu12.5-amd-amdhsa-unknown-gfx12-5-generic
.amdgcn_target "amdgpu12.5-amd-amdhsa-unknown-gfx1251"

//--- amdpal.s
// AMDPAL: .amd_amdgpu_isa "amdgpu12.50-amd-amdpal-unknown-gfx1250"
.amd_amdgpu_isa "amdgpu12.50-amd-amdpal-unknown-"

//--- amdpal-generic-subarch-cpu.s
// FIXME: This should be accepted and passed through as-is
// AMDPAL-GENERIC-SUBARCH: error: .amd_amdgpu_isa directive's target id amdgpu12.5-amd-amdpal-unknown-gfx1251 does not match the specified target id amdgpu12.5-amd-amdpal-unknown-gfx12-5-generic
.amd_amdgpu_isa "amdgpu12.5-amd-amdpal-unknown-gfx1251"

//--- cpu.s
// A processor covered by the subarch is accepted.
.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"

//--- subarch-mismatch.s
// A target id whose subarch conflicts with the triple is rejected: the subarch
// resolves to a processor that is not valid for the triple's subarch.
// ERR: {{.*}}: error: target id 'amdgpu9.00-amd-amdhsa-unknown-' specifies a processor that is not valid for subarch 'amdgpu12.50'
.amdgcn_target "amdgpu9.00-amd-amdhsa-unknown-"

//--- cpu-not-covered.s
// A processor not covered by the triple's subarch is rejected.
// NOTCOVERED: {{.*}}: error: target id 'amdgpu12.50-amd-amdhsa-unknown-gfx900' specifies a processor that is not valid for subarch 'amdgpu12.50'
.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx900"

//--- cpu-not-covered-same-major.s
// A sibling processor sharing the major subarch (gfx1251 and gfx1250 are both
// gfx12.5) is still not covered by a specific subarch (amdgpu12.50).
// NOTCOVERED-SAME-MAJOR: {{.*}}: error: target id 'amdgpu12.50-amd-amdhsa-unknown-gfx1251' specifies a processor that is not valid for subarch 'amdgpu12.50'
.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1251"

//--- isa-cpu-not-covered.s
// The same check applies to .amd_amdgpu_isa.
// ISA-NOTCOVERED: {{.*}}: error: target id 'amdgpu12.50-amd-amdpal-unknown-gfx900' specifies a processor that is not valid for subarch 'amdgpu12.50'
.amd_amdgpu_isa "amdgpu12.50-amd-amdpal-unknown-gfx900"
