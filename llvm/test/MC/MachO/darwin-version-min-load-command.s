// RUN: llvm-mc -triple x86_64-apple-macosx10.10.0 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s
// RUN: llvm-mc -triple x86_64-apple-ios8.0.0 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-IOS
// RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-DARWIN
// RUN: llvm-mc -triple x86_64-apple-ios13.0-macabi %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-MACCATALYST

// RUN: llvm-mc -triple x86_64-apple-macos10.14 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-MACOS
// RUN: llvm-mc -triple x86_64-apple-ios12 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOS
// RUN: llvm-mc -triple x86_64-apple-tvos12 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-TVOS
// RUN: llvm-mc -triple x86_64-apple-watchos5 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-WATCHOS
// RUN: llvm-mc -triple x86_64-apple-ios12-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM
// RUN: llvm-mc -triple x86_64-apple-tvos12-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-TVOSSIM
// RUN: llvm-mc -triple x86_64-apple-watchos5-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-WATCHOSSIM

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


// RUN: llvm-mc -triple x86_64-apple-watchos1.0.0 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-WATCHOS
// RUN: llvm-mc -triple x86_64-apple-tvos8.0.0 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-TVOS
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

// CHECK-MACCATALYST:           cmd LC_BUILD_VERSION
// CHECK-MACCATALYST-NEXT:  cmdsize 24
// CHECK-MACCATALYST-NEXT: platform macCatalyst
// CHECK-MACCATALYST-NEXT:      sdk n/a
// CHECK-MACCATALYST-NEXT:    minos 13.0
// CHECK-MACCATALYST-NEXT:   ntools 0

// CHECK-BUILD-MACOS:           cmd LC_BUILD_VERSION
// CHECK-BUILD-MACOS-NEXT:  cmdsize 24
// CHECK-BUILD-MACOS-NEXT: platform macos
// CHECK-BUILD-MACOS-NEXT:      sdk n/a
// CHECK-BUILD-MACOS-NEXT:    minos 10.14
// CHECK-BUILD-MACOS-NEXT:   ntools 0
// CHECK-BUILD-MACOS-NOT: LC_VERSION_MIN

// CHECK-BUILD-IOS:           cmd LC_BUILD_VERSION
// CHECK-BUILD-IOS-NEXT:  cmdsize 24
// CHECK-BUILD-IOS-NEXT: platform ios
// CHECK-BUILD-IOS-NEXT:      sdk n/a
// CHECK-BUILD-IOS-NEXT:    minos 12.0
// CHECK-BUILD-IOS-NEXT:   ntools 0
// CHECK-BUILD-IOS-NOT: LC_VERSION_MIN

// CHECK-BUILD-TVOS:           cmd LC_BUILD_VERSION
// CHECK-BUILD-TVOS-NEXT:  cmdsize 24
// CHECK-BUILD-TVOS-NEXT: platform tvos
// CHECK-BUILD-TVOS-NEXT:      sdk n/a
// CHECK-BUILD-TVOS-NEXT:    minos 12.0
// CHECK-BUILD-TVOS-NEXT:   ntools 0
// CHECK-BUILD-TVOS-NOT: LC_VERSION_MIN

// CHECK-BUILD-WATCHOS:           cmd LC_BUILD_VERSION
// CHECK-BUILD-WATCHOS-NEXT:  cmdsize 24
// CHECK-BUILD-WATCHOS-NEXT: platform watchos
// CHECK-BUILD-WATCHOS-NEXT:      sdk n/a
// CHECK-BUILD-WATCHOS-NEXT:    minos 5.0
// CHECK-BUILD-WATCHOS-NEXT:   ntools 0
// CHECK-BUILD-WATCHOS-NOT: LC_VERSION_MIN

// CHECK-BUILD-IOSSIM:           cmd LC_BUILD_VERSION
// CHECK-BUILD-IOSSIM-NEXT:  cmdsize 24
// CHECK-BUILD-IOSSIM-NEXT: platform iossim
// CHECK-BUILD-IOSSIM-NEXT:      sdk n/a
// CHECK-BUILD-IOSSIM-NEXT:    minos 12.0
// CHECK-BUILD-IOSSIM-NEXT:   ntools 0
// CHECK-BUILD-IOSSIM-NOT: LC_VERSION_MIN

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

// CHECK-BUILD-TVOSSIM:           cmd LC_BUILD_VERSION
// CHECK-BUILD-TVOSSIM-NEXT:  cmdsize 24
// CHECK-BUILD-TVOSSIM-NEXT: platform tvossim
// CHECK-BUILD-TVOSSIM-NEXT:      sdk n/a
// CHECK-BUILD-TVOSSIM-NEXT:    minos 12.0
// CHECK-BUILD-TVOSSIM-NEXT:   ntools 0
// CHECK-BUILD-TVOSSIM-NOT: LC_VERSION_MIN

// CHECK-BUILD-TVOSSIM2:           cmd LC_BUILD_VERSION
// CHECK-BUILD-TVOSSIM2-NEXT:  cmdsize 24
// CHECK-BUILD-TVOSSIM2-NEXT: platform tvossim
// CHECK-BUILD-TVOSSIM2-NEXT:      sdk n/a
// CHECK-BUILD-TVOSSIM2-NEXT:    minos 14.0
// CHECK-BUILD-TVOSSIM2-NEXT:   ntools 0
// CHECK-BUILD-TVOSSIM2-NOT: LC_VERSION_MIN

// CHECK-BUILD-WATCHOSSIM:           cmd LC_BUILD_VERSION
// CHECK-BUILD-WATCHOSSIM-NEXT:  cmdsize 24
// CHECK-BUILD-WATCHOSSIM-NEXT: platform watchossim
// CHECK-BUILD-WATCHOSSIM-NEXT:      sdk n/a
// CHECK-BUILD-WATCHOSSIM-NEXT:    minos 5.0
// CHECK-BUILD-WATCHOSSIM-NEXT:   ntools 0
// CHECK-BUILD-WATCHOSSIM-NOT: LC_VERSION_MIN

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
