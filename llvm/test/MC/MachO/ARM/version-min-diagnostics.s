// RUN: not llvm-mc -triple i386-apple-darwin %s 2> %t
// RUN: FileCheck %s < %t
// RUN: not llvm-mc -triple x86_64-apple-darwin %s 2> %t
// RUN: FileCheck %s < %t
// RUN: not llvm-mc -triple armv7-apple-ios %s 2> %t
// RUN: FileCheck %s < %t

.ios_version_min 5,2,257
.ios_version_min 5,256,1
.ios_version_min 5,-1,1
.ios_version_min 0,1,1
.ios_version_min 70000,1
.macosx_version_min 99,2,257
.macosx_version_min 50,256,1
.macosx_version_min 10,-1,1
.macosx_version_min 0,1,1
.macosx_version_min 70000,1
.tvos_version_min 99,2,257
.tvos_version_min 50,256,1
.tvos_version_min 10,-1,1
.tvos_version_min 0,1,1
.tvos_version_min 70000,1
.watchos_version_min 99,2,257
.watchos_version_min 50,256,1
.watchos_version_min 10,-1,1
.watchos_version_min 0,1,1
.watchos_version_min 70000,1


// CHECK: error: invalid OS update version number
// CHECK: .ios_version_min 5,2,257
// CHECK:                      ^
// CHECK: error: invalid OS minor version number
// CHECK: .ios_version_min 5,256,1
// CHECK:                    ^
// CHECK: error: invalid OS minor version number
// CHECK: .ios_version_min 5,-1,1
// CHECK:                    ^
// CHECK: error: invalid OS major version number
// CHECK: .ios_version_min 0,1,1
// CHECK:                  ^
// CHECK: error: invalid OS major version number
// CHECK: .ios_version_min 70000,1
// CHECK:                  ^
// CHECK: error: invalid OS update version number
// CHECK: .macosx_version_min 99,2,257
// CHECK:                          ^
// CHECK: error: invalid OS minor version number
// CHECK: .macosx_version_min 50,256,1
// CHECK:                        ^
// CHECK: error: invalid OS minor version number
// CHECK: .macosx_version_min 10,-1,1
// CHECK:                        ^
// CHECK: error: invalid OS major version number
// CHECK: .macosx_version_min 0,1,1
// CHECK:                     ^
// CHECK: error: invalid OS major version number
// CHECK: .macosx_version_min 70000,1
// CHECK:                     ^
// CHECK: error: invalid OS update version number
// CHECK: .tvos_version_min 99,2,257
// CHECK:                          ^
// CHECK: error: invalid OS minor version number
// CHECK: .tvos_version_min 50,256,1
// CHECK:                        ^
// CHECK: error: invalid OS minor version number
// CHECK: .tvos_version_min 10,-1,1
// CHECK:                        ^
// CHECK: error: invalid OS major version number
// CHECK: .tvos_version_min 0,1,1
// CHECK:                     ^
// CHECK: error: invalid OS major version number
// CHECK: .tvos_version_min 70000,1
// CHECK:                     ^
// CHECK: error: invalid OS update version number
// CHECK: .watchos_version_min 99,2,257
// CHECK:                          ^
// CHECK: error: invalid OS minor version number
// CHECK: .watchos_version_min 50,256,1
// CHECK:                        ^
// CHECK: error: invalid OS minor version number
// CHECK: .watchos_version_min 10,-1,1
// CHECK:                        ^
// CHECK: error: invalid OS major version number
// CHECK: .watchos_version_min 0,1,1
// CHECK:                     ^
// CHECK: error: invalid OS major version number
// CHECK: .watchos_version_min 70000,1
// CHECK:                     ^
