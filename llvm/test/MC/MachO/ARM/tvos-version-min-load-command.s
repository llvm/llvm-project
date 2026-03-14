// RUN: llvm-mc -triple armv7-apple-tvos %s -filetype=obj -o - | llvm-readobj --macho-version-min - | FileCheck %s


// Test the formation of the version-min load command in the MachO.
// use a nonsense but well formed version.
.tvos_version_min 99,8,7

// CHECK: MinVersion {
// CHECK-NEXT:   Cmd: LC_VERSION_MIN_TVOS
// CHECK-NEXT:   Size: 16
// CHECK-NEXT:   Version: 99.8.7
// CHECK-NEXT:   SDK: n/a
// CHECK-NEXT: }
