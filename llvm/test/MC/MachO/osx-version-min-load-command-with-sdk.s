// RUN: llvm-mc -triple x86_64-apple-macos %s -filetype=obj -o - | llvm-readobj --macho-version-min | FileCheck %s

// Test the formation of the sdk_version component of the version load
// command in the MachO.
.macosx_version_min 10,13,2 sdk_version 10,14

// CHECK: MinVersion {
// CHECK:   Cmd: LC_VERSION_MIN_MACOSX
// CHECK:   Size: 16
// CHECK:   Version: 10.13.2
// CHECK:   SDK: 10.14
// CHECK: }
