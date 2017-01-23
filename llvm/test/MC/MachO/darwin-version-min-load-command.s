// RUN: llvm-mc -triple x86_64-apple-macosx10.10.0 %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s
// RUN: llvm-mc -triple x86_64-apple-ios8.0.0 %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s --check-prefix=CHECK-IOS
// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s --check-prefix=CHECK-DARWIN

// Test version-min load command should be inferred from triple and should always be generated on Darwin
// CHECK: Load command
// CHECK:       cmd LC_VERSION_MIN_MACOSX
// CHECK:   cmdsize 16
// CHECK:   version 10.10

// CHECK-IOS: Load command
// CHECK-IOS:       cmd LC_VERSION_MIN_IPHONEOS
// CHECK-IOS:   cmdsize 16
// CHECK-IOS:   version 8.0

// CHECK-DARWIN-NOT: LC_VERSION_MIN


// RUN: llvm-mc -triple x86_64-apple-watchos1.0.0 %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s --check-prefix=CHECK-WATCHOS
// RUN: llvm-mc -triple x86_64-apple-tvos8.0.0 %s -filetype=obj -o - | llvm-objdump -macho -private-headers - | FileCheck %s --check-prefix=CHECK-TVOS
// CHECK-WATCHOS: Load command
// CHECK-WATCHOS:           cmd LC_VERSION_MIN_WATCHOS
// CHECK-WATCHOS-NEXT:   cmdsize 16
// CHECK-WATCHOS-NEXT:   version 1.0

// CHECK-TVOS:            cmd LC_VERSION_MIN_TVOS
// CHECK-TVOS-NEXT:   cmdsize 16
// CHECK-TVOS-NEXT:   version 8.0

// CHECK-BRIDGEOS:            cmd LC_BUILD_VERSION
// CHECK-BRIDGEOS-NEXT:   cmdsize 24
// CHECK-BRIDGEOS-NEXT:  platform bridgeos
// CHECK-BRIDGEOS-NEXT:       sdk n/a
// CHECK-BRIDGEOS-NEXT:     minos 2.0
// CHECK-BRIDGEOS-NEXT:    ntools 0
