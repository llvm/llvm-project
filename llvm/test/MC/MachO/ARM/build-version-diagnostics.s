// RUN: not llvm-mc -triple x86_64-apple-tvos %s 2>&1 | FileCheck %s

.build_version tvos,65535,0,255
// CHECK-NOT: build-version-diagnostics.s:[[@LINE-1]]

.build_version macos,1,2,3
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:1: warning: .build_version macos used while targeting tvos
// CHECK: build-version-diagnostics.s:[[@LINE-2]]:1: warning: overriding previous version directive

.build_version 1,2,3
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:16: error: platform name expected

.build_version macos 1,2
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:22: error: version number required, comma expected

.build_version noos,1,2
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:16: error: unknown platform name

.build_version macos,a
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:22: error: invalid OS major version number, integer expected

.build_version macos,0,1
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:22: error: invalid OS major version number

.build_version macos,-1,1
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:22: error: invalid OS major version number

.build_version macos,65536,1
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:22: error: invalid OS major version number

.build_version ios,10 0
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:23: error: OS minor version number required, comma expected

.build_version ios,10,
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:23: error: invalid OS minor version number, integer expected

.build_version ios,10,-1
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:23: error: invalid OS minor version number

.build_version ios,10,256
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:23: error: invalid OS minor version number

.build_version watchos,10,0 a
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:29: error: invalid OS update specifier, comma expected

.build_version watchos,10,0 ,
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:30: error: invalid OS update version number, integer expected

.build_version ios,10,0,-1
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:25: error: invalid OS update version number

.build_version ios,10,0,256
// CHECK: build-version-diagnostics.s:[[@LINE-1]]:25: error: invalid OS update version number

.build_version ios,10,0,0,
// CHECK: build-version-diagnostics.s:[[#@LINE-1]]:26: error: expected newline in '.build_version' directive
