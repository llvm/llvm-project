// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | llvm-readobj --macho-version-min - | FileCheck %s

// Test the formation of the version-min load command in the MachO.
// use a nonsense but well formed version.
.macosx_version_min 25,3,1

// CHECK: File: <stdin>
// CHECK: Format: Mach-O 64-bit x86-64
// CHECK: Arch: x86_64
// CHECK: AddressSize: 64bit
// CHECK: MinVersion {
// CHECK:   Cmd: LC_VERSION_MIN_MACOSX
// CHECK:   Size: 16
// CHECK:   Version: 25.3.1
// CHECK:   SDK: n/a
// CHECK: }
