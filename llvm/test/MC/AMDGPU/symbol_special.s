// RUN: llvm-mc -arch=amdgcn -mcpu=bonaire %s | FileCheck %s --check-prefix=BONAIRE
// RUN: llvm-mc -arch=amdgcn -mcpu=hawaii %s | FileCheck %s --check-prefix=HAWAII
// RUN: llvm-mc -arch=amdgcn -mcpu=tonga %s | FileCheck %s --check-prefix=TONGA
// RUN: llvm-mc -arch=amdgcn -mcpu=fiji %s | FileCheck %s --check-prefix=FIJI

.if .option.machine_version_major == 0
.byte 0
.elseif .option.machine_version_major == 7
.byte 7
.elseif .option.machine_version_major == 8
.byte 8
.else
.error "major unknown"
.endif
// BONAIRE: .byte 7
// HAWAII: .byte 7
// TONGA: .byte 8
// FIJI: .byte 8

.if .option.machine_version_minor == 0
.byte 0
.else
.error "minor unknown"
.endif
// BONAIRE: .byte 0
// HAWAII: .byte 0
// TONGA: .byte 0
// FIJI: .byte 0

.if .option.machine_version_stepping == 0
.byte 0
.elseif .option.machine_version_stepping == 1
.byte 1
.elseif .option.machine_version_stepping == 3
.byte 3
.else
.error "stepping unknown"
.endif
// BONAIRE: .byte 0
// HAWAII: .byte 1
// TONGA: .byte 0
// FIJI: .byte 3

v_add_f32 v0, v0, v[.option.machine_version_major]
// BONAIRE: v_add_f32_e32 v0, v0, v7
// HAWAII: v_add_f32_e32 v0, v0, v7
// TONGA: v_add_f32_e32 v0, v0, v8
// FIJI: v_add_f32_e32 v0, v0, v8
