// RUN: llvm-mc -arch=amdgcn -mcpu=tahiti      %s | FileCheck %s --check-prefix=SI
// RUN: llvm-mc -arch=amdgcn -mcpu=bonaire %s | FileCheck %s --check-prefix=BONAIRE
// RUN: llvm-mc -arch=amdgcn -mcpu=hawaii %s | FileCheck %s --check-prefix=HAWAII
// RUN: llvm-mc -arch=amdgcn -mcpu=kabini  %s | FileCheck %s --check-prefix=KABINI
// RUN: llvm-mc -arch=amdgcn -mcpu=iceland %s | FileCheck %s --check-prefix=ICELAND
// RUN: llvm-mc -arch=amdgcn -mcpu=carrizo %s | FileCheck %s --check-prefix=CARRIZO
// RUN: llvm-mc -arch=amdgcn -mcpu=tonga %s | FileCheck %s --check-prefix=TONGA
// RUN: llvm-mc -arch=amdgcn -mcpu=fiji %s | FileCheck %s --check-prefix=FIJI
// RUN: llvm-mc -arch=amdgcn -mcpu=stoney  %s | FileCheck %s --check-prefix=STONEY

.byte .option.machine_version_major
// SI: .byte 6
// BONAIRE: .byte 7
// HAWAII: .byte 7
// KABINI: .byte 7
// ICELAND: .byte 8
// CARRIZO: .byte 8
// TONGA: .byte 8
// FIJI: .byte 8
// STONEY: .byte 8

.byte .option.machine_version_minor
// SI: .byte 0
// BONAIRE: .byte 0
// HAWAII: .byte 0
// KABINI: .byte 0
// ICELAND: .byte 0
// CARRIZO: .byte 0
// TONGA: .byte 0
// FIJI: .byte 0
// STONEY: .byte 1

.byte .option.machine_version_stepping
// SI: .byte 0
// BONAIRE: .byte 4
// HAWAII: .byte 1
// KABINI: .byte 3
// ICELAND: .byte 2
// CARRIZO: .byte 1
// TONGA: .byte 2
// FIJI: .byte 3
// STONEY: .byte 0
