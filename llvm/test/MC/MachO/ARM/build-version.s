// RUN: llvm-mc -triple x86_64-apple-macos %s | FileCheck %s

.build_version macos,1,2
// CHECK: .build_version macos, 1, 2

.build_version macos,1,2,0
// CHECK: .build_version macos, 1, 2

.build_version macos,3,4,5
// CHECK: .build_version macos, 3, 4, 5

.build_version ios,6,7
// CHECK: .build_version ios, 6, 7

.build_version tvos,8,9
// CHECK: .build_version tvos, 8, 9

.build_version watchos,10,11
// CHECK: .build_version watchos, 10, 11

.build_version xros,1, 0 
// CHECK: .build_version xros, 1, 0 
