// RUN: not llvm-mc -triple x86_64-apple-macos %s -o - 2>&1 | FileCheck %s

.macosx_version_min 10,13,2 sdk_version 10
// CHECK: SDK minor version number required, comma expected

.macosx_version_min 10,13,2 sdk_version 10,
// CHECK: invalid SDK minor version number, integer expected

.macosx_version_min 10,13,2 sdk_version 10
// CHECK: SDK minor version number required, comma expected
