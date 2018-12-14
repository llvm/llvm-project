// RUN: llvm-mc -triple x86_64-apple-macos %s | FileCheck %s

.build_version macos,3,4,5 sdk_version 10,14
// CHECK: .build_version macos, 3, 4, 5 sdk_version 10, 14

.build_version ios,6,7 sdk_version 6,1,0
// CHECK: .build_version ios, 6, 7 sdk_version 6, 1, 0

.build_version tvos,8,9 sdk_version 9,0,10
// CHECK: .build_version tvos, 8, 9 sdk_version 9, 0, 10

.build_version watchos,10,11 sdk_version 10,11
// CHECK: .build_version watchos, 10, 11 sdk_version 10, 11
