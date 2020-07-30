// RUN: llvm-mc -triple arm64-apple-macos10.10.2 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-MACOS-ARM64
// RUN: llvm-mc -triple arm64-apple-macos11 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-MACOS-ARM64
// RUN: llvm-mc -triple arm64-apple-macos11.1 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-MACOS-ARM64_1
// RUN: llvm-mc -triple arm64-apple-ios13.0-macabi %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-MACCATALYST-ARM64
// RUN: llvm-mc -triple arm64-apple-ios14.1-macabi %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-MACCATALYST-ARM64_1

// RUN: llvm-mc -triple arm64-apple-ios10-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM2
// RUN: llvm-mc -triple arm64-apple-ios13-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM2
// RUN: llvm-mc -triple arm64-apple-ios14-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM2
// RUN: llvm-mc -triple arm64-apple-ios14.1-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM3
// RUN: llvm-mc -triple arm64-apple-tvos10-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-TVOSSIM2
// RUN: llvm-mc -triple arm64-apple-watchos3-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-WATCHOSSIM2

// CHECK-BUILD-IOSSIM2:           cmd LC_BUILD_VERSION
// CHECK-BUILD-IOSSIM2-NEXT:  cmdsize 24
// CHECK-BUILD-IOSSIM2-NEXT: platform iossim
// CHECK-BUILD-IOSSIM2-NEXT:      sdk n/a
// CHECK-BUILD-IOSSIM2-NEXT:    minos 14.0
// CHECK-BUILD-IOSSIM2-NEXT:   ntools 0
// CHECK-BUILD-IOSSIM2-NOT: LC_VERSION_MIN

// CHECK-BUILD-IOSSIM3:           cmd LC_BUILD_VERSION
// CHECK-BUILD-IOSSIM3-NEXT:  cmdsize 24
// CHECK-BUILD-IOSSIM3-NEXT: platform iossim
// CHECK-BUILD-IOSSIM3-NEXT:      sdk n/a
// CHECK-BUILD-IOSSIM3-NEXT:    minos 14.1
// CHECK-BUILD-IOSSIM3-NEXT:   ntools 0
// CHECK-BUILD-IOSSIM3-NOT: LC_VERSION_MIN

// CHECK-BUILD-TVOSSIM2:           cmd LC_BUILD_VERSION
// CHECK-BUILD-TVOSSIM2-NEXT:  cmdsize 24
// CHECK-BUILD-TVOSSIM2-NEXT: platform tvossim
// CHECK-BUILD-TVOSSIM2-NEXT:      sdk n/a
// CHECK-BUILD-TVOSSIM2-NEXT:    minos 14.0
// CHECK-BUILD-TVOSSIM2-NEXT:   ntools 0
// CHECK-BUILD-TVOSSIM2-NOT: LC_VERSION_MIN

// CHECK-BUILD-WATCHOSSIM2:           cmd LC_BUILD_VERSION
// CHECK-BUILD-WATCHOSSIM2-NEXT:  cmdsize 24
// CHECK-BUILD-WATCHOSSIM2-NEXT: platform watchossim
// CHECK-BUILD-WATCHOSSIM2-NEXT:      sdk n/a
// CHECK-BUILD-WATCHOSSIM2-NEXT:    minos 7.0
// CHECK-BUILD-WATCHOSSIM2-NEXT:   ntools 0
// CHECK-BUILD-WATCHOSSIM2-NOT: LC_VERSION_MIN

// CHECK-BUILD-MACOS-ARM64:           cmd LC_BUILD_VERSION
// CHECK-BUILD-MACOS-ARM64-NEXT:  cmdsize 24
// CHECK-BUILD-MACOS-ARM64-NEXT: platform macos
// CHECK-BUILD-MACOS-ARM64-NEXT:      sdk n/a
// CHECK-BUILD-MACOS-ARM64-NEXT:    minos 11.0
// CHECK-BUILD-MACOS-ARM64-NEXT:   ntools 0
// CHECK-BUILD-MACOS-ARM64-NOT: LC_VERSION_MIN

// CHECK-BUILD-MACOS-ARM64_1:           cmd LC_BUILD_VERSION
// CHECK-BUILD-MACOS-ARM64_1-NEXT:  cmdsize 24
// CHECK-BUILD-MACOS-ARM64_1-NEXT: platform macos
// CHECK-BUILD-MACOS-ARM64_1-NEXT:      sdk n/a
// CHECK-BUILD-MACOS-ARM64_1-NEXT:    minos 11.1
// CHECK-BUILD-MACOS-ARM64_1-NEXT:   ntools 0

// CHECK-MACCATALYST-ARM64:           cmd LC_BUILD_VERSION
// CHECK-MACCATALYST-ARM64-NEXT:  cmdsize 24
// CHECK-MACCATALYST-ARM64-NEXT: platform macCatalyst
// CHECK-MACCATALYST-ARM64-NEXT:      sdk n/a
// CHECK-MACCATALYST-ARM64-NEXT:    minos 14.0
// CHECK-MACCATALYST-ARM64-NEXT:   ntools 0

// CHECK-MACCATALYST-ARM64_1:           cmd LC_BUILD_VERSION
// CHECK-MACCATALYST-ARM64_1-NEXT:  cmdsize 24
// CHECK-MACCATALYST-ARM64_1-NEXT: platform macCatalyst
// CHECK-MACCATALYST-ARM64_1-NEXT:      sdk n/a
// CHECK-MACCATALYST-ARM64_1-NEXT:    minos 14.1
// CHECK-MACCATALYST-ARM64_1-NEXT:   ntools 0
