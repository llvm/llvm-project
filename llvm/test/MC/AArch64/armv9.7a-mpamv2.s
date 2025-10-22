// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mpamv2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mpamv2 < %s \
// RUN:        | llvm-objdump -d --mattr=+mpamv2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mpamv2 < %s \
// RUN:        | llvm-objdump -d --mattr=-mpamv2 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mpamv2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+mpamv2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// Armv9.7-A FEAT_MPAMV2 Extensions
//------------------------------------------------------------------------------

msr MPAMCTL_EL1, x0
// CHECK-INST:    msr MPAMCTL_EL1, x0
// CHECK-ENCODING: [0x40,0xa5,0x18,0xd5]
// CHECK-UNKNOWN: d518a540

msr MPAMCTL_EL12, x0
// CHECK-INST:   msr MPAMCTL_EL12, x0
// CHECK-ENCODING: [0x40,0xa5,0x1d,0xd5]
// CHECK-UNKNOWN: d51da540

msr MPAMCTL_EL2, x0
// CHECK-INST:    msr     MPAMCTL_EL2, x0
// CHECK-ENCODING: [0x40,0xa5,0x1c,0xd5]
// CHECK-UNKNOWN: d51ca540

msr MPAMCTL_EL3, x0
// CHECK-INST:    msr     MPAMCTL_EL3, x0
// CHECK-ENCODING: [0x40,0xa5,0x1e,0xd5]
// CHECK-UNKNOWN: d51ea540

msr MPAMVIDCR_EL2, x0
// CHECK-INST:    msr     MPAMVIDCR_EL2, x0
// CHECK-ENCODING: [0x00,0xa7,0x1c,0xd5]
// CHECK-UNKNOWN: d51ca700

msr MPAMVIDSR_EL2, x0
// CHECK-INST:    msr     MPAMVIDSR_EL2, x0
// CHECK-ENCODING: [0x20,0xa7,0x1c,0xd5]
// CHECK-UNKNOWN: d51ca720

msr MPAMVIDSR_EL3, x0
// CHECK-INST:    msr     MPAMVIDSR_EL3, x0
// CHECK-ENCODING: [0x20,0xa7,0x1e,0xd5]
// CHECK-UNKNOWN: d51ea720


mrs x0, MPAMCTL_EL1
// CHECK-INST:        mrs     x0, MPAMCTL_EL1
// CHECK-ENCODING: [0x40,0xa5,0x38,0xd5]
// CHECK-UNKNOWN: d538a540

mrs x0, MPAMCTL_EL12
// CHECK-INST:   mrs     x0, MPAMCTL_EL12
// CHECK-ENCODING: [0x40,0xa5,0x3d,0xd5]
// CHECK-UNKNOWN: d53da540

mrs x0, MPAMCTL_EL2
// CHECK-INST:   mrs     x0, MPAMCTL_EL2
// CHECK-ENCODING: [0x40,0xa5,0x3c,0xd5]
// CHECK-UNKNOWN: d53ca540

mrs x0, MPAMCTL_EL3
// CHECK-INST:   mrs     x0, MPAMCTL_EL3
// CHECK-ENCODING: [0x40,0xa5,0x3e,0xd5]
// CHECK-UNKNOWN: d53ea540

mrs x0, MPAMVIDCR_EL2
// CHECK-INST:   mrs     x0, MPAMVIDCR_EL2
// CHECK-ENCODING: [0x00,0xa7,0x3c,0xd5]
// CHECK-UNKNOWN: d53ca700

mrs x0, MPAMVIDSR_EL2
// CHECK-INST:   mrs     x0, MPAMVIDSR_EL2
// CHECK-ENCODING: [0x20,0xa7,0x3c,0xd5]
// CHECK-UNKNOWN: d53ca720

mrs x0, MPAMVIDSR_EL3
// CHECK-INST:   mrs     x0, MPAMVIDSR_EL3
// CHECK-ENCODING: [0x20,0xa7,0x3e,0xd5]
// CHECK-UNKNOWN: d53ea720


//------------------------------------------------------------------------------
// Armv9.7-A FEAT_MPAMV2_VID Extensions
//------------------------------------------------------------------------------

mlbi vmalle1
// CHECK-INST:    mlbi vmalle1
// CHECK-ENCODING: [0xbf,0x70,0x0c,0xd5]
// CHECK-UNKNOWN: d50c70bf sys	#4, c7, c0, #5
// CHECK-ERROR: error: MLBI VMALLE1 requires: mpamv2

mlbi vpide1, x0
// CHECK-INST:    mlbi vpide1, x0
// CHECK-ENCODING: [0xc0,0x70,0x0c,0xd5]
// CHECK-UNKNOWN: d50c70c0 sys	#4, c7, c0, #6, x0
// CHECK-ERROR: error: MLBI VPIDE1 requires: mpamv2

mlbi vpmge1, x0
// CHECK-INST:    mlbi vpmge1, x0
// CHECK-ENCODING: [0xe0,0x70,0x0c,0xd5]
// CHECK-UNKNOWN: d50c70e0 sys	#4, c7, c0, #7, x0
// CHECK-ERROR: error: MLBI VPMGE1 requires: mpamv2

// Check that invalid encodings are rendered as SYS aliases
// [0x9f,0x70,0x0c,0xd5] -> mlbi alle1
// [0x9e,0x70,0x0c,0xd5] -> sys #4, c7, c0, #4, x30

mlbi alle1
// CHECK-INST:    mlbi alle1
// CHECK-ENCODING: [0x9f,0x70,0x0c,0xd5]
// CHECK-UNKNOWN: d50c709f sys	#4, c7, c0, #4
// CHECK-ERROR: error: MLBI ALLE1 requires: mpamv2

sys #4, c7, c0, #4, x30
// CHECK-INST: sys #4, c7, c0, #4, x30
// CHECK-ENCODING: [0x9e,0x70,0x0c,0xd5]
// CHECK-UNKNOWN: d50c709e sys #4, c7, c0, #4, x30
