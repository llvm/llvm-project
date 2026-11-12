// RUN: llvm-mc -triple armv7-apple-ios %s -filetype=obj -o - | llvm-readobj --macho-version-min - | FileCheck %s

// Test the formation of the version-min load command in the MachO.
// use a nonsense but well formed version.
.ios_version_min 99,8,7

// CHECK: File: <stdin>
// CHECK: Format: Mach-O arm
// CHECK: Arch: arm
// CHECK: AddressSize: 32bit
// CHECK: MinVersion {
// CHECK:   Cmd: LC_VERSION_MIN_IPHONEOS
// CHECK:   Size: 16
// CHECK:   Version: 99.8.7
// CHECK:   SDK: n/a
// CHECK: }
