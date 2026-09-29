// RUN: llvm-mc -triple i386-apple-darwin %s | FileCheck %s
// RUN: llvm-mc -triple x86_64-apple-darwin %s | FileCheck %s
// RUN: llvm-mc -triple armv7s-apple-ios %s | FileCheck %s

// Test the parsing of well-formed version-min directives.

.ios_version_min 5,2,0
.ios_version_min 3,2,1
.ios_version_min 5,0

// CHECK: .ios_version_min 5, 2
// CHECK: .ios_version_min 3, 2, 1
// CHECK: .ios_version_min 5, 0

.macosx_version_min 10,2,0
.macosx_version_min 10,8,1
.macosx_version_min 2,0

// CHECK: .macosx_version_min 10, 2
// CHECK: .macosx_version_min 10, 8, 1
// CHECK: .macosx_version_min 2, 0

.tvos_version_min 5,2,0
.tvos_version_min 3,2,1
.tvos_version_min 5,0

// CHECK: .tvos_version_min 5, 2
// CHECK: .tvos_version_min 3, 2, 1
// CHECK: .tvos_version_min 5, 0

.watchos_version_min 5,2,0
.watchos_version_min 3,2,1
.watchos_version_min 5,0

// CHECK: .watchos_version_min 5, 2
// CHECK: .watchos_version_min 3, 2, 1
// CHECK: .watchos_version_min 5, 0
