// Regression test for hexagonAttrToFeatureString in
/// llvm/lib/Object/ELFObjectFile.cpp.
///

r0 = add(r1,r2)

// RUN: llvm-mc -triple=hexagon --mcpu=hexagonv5 %s \
// RUN:   -filetype=obj --hexagon-add-build-attributes -o %t.v5.o
// RUN: llvm-readelf -A %t.v5.o | FileCheck %s --check-prefix=V5
// RUN: llvm-objdump -d %t.v5.o | FileCheck %s --check-prefix=DIS

// RUN: llvm-mc -triple=hexagon --mcpu=hexagonv55 %s \
// RUN:   -filetype=obj --hexagon-add-build-attributes -o %t.v55.o
// RUN: llvm-readelf -A %t.v55.o | FileCheck %s --check-prefix=V55
// RUN: llvm-objdump -d %t.v55.o | FileCheck %s --check-prefix=DIS

// RUN: llvm-mc -triple=hexagon --mcpu=hexagonv68 %s \
// RUN:   -filetype=obj --hexagon-add-build-attributes -o %t.v68.o
// RUN: llvm-readelf -A %t.v68.o | FileCheck %s --check-prefix=V68
// RUN: llvm-objdump -d %t.v68.o | FileCheck %s --check-prefix=DIS

// RUN: llvm-mc -triple=hexagon --mcpu=hexagonv75 %s \
// RUN:   -filetype=obj --hexagon-add-build-attributes -o %t.v75.o
// RUN: llvm-readelf -A %t.v75.o | FileCheck %s --check-prefix=V75
// RUN: llvm-objdump -d %t.v75.o | FileCheck %s --check-prefix=DIS

/// Test HVX arch attribute mapping. llvm-objdump should pick up hvx features
/// from build attributes and disassemble HVX instructions correctly.
// RUN: llvm-mc -triple=hexagon --mcpu=hexagonv68 -mhvx %s \
// RUN:   -filetype=obj --hexagon-add-build-attributes -o %t.v68.hvx.o
// RUN: llvm-readelf -A %t.v68.hvx.o | FileCheck %s --check-prefix=HVX68
// RUN: llvm-objdump -d %t.v68.hvx.o | FileCheck %s --check-prefix=DIS

/// Each readelf invocation should record the matching arch tag value.
// V5: TagName: arch
// V5-NEXT: Value: 5
// V55: TagName: arch
// V55-NEXT: Value: 55
// V68: TagName: arch
// V68-NEXT: Value: 68
// V75: TagName: arch
// V75-NEXT: Value: 75

/// HVX arch attribute should be recorded when -mhvx is used.
// HVX68: TagName: hvx_arch
// HVX68-NEXT: Value: 68

/// llvm-objdump should disassemble the instruction correctly for every
/// arch version, with no <unknown> fallback.
// DIS:      <.text>:
// DIS-NEXT: r0 = add(r1,r2)
// DIS-NOT:  <unknown>
