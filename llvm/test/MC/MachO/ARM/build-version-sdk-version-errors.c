// RUN: not llvm-mc -triple x86_64-apple-macos %s 2>&1 | FileCheck %s

.build_version macos,3,4,5 sdk_version
// CHECK: invalid SDK major version number, integer expected

.build_version macos,3,4,5 sdk_version 10
// CHECK: SDK minor version number required, comma expected

.build_version macos,3,4,5 sdk_version 10,
// CHECK: invalid SDK minor version number, integer expected

.build_version macos,3,4,5 sdk_version 10,1,
// CHECK: invalid SDK subminor version number, integer expected

.build_version macos,3,4,5 sdk_version 10,10000
// CHECK: invalid SDK minor version number

.build_version macos,3,4,5 sdk_version 10,255,10000
// CHECK: invalid SDK subminor version number
