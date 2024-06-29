// RUN: llvm-mc -triple arm64-apple-macos10.10.2 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-MACOS-ARM64
// RUN: llvm-mc -triple arm64e-apple-macos10.10 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-MACOS-ARM64
// RUN: llvm-mc -triple arm64-apple-macos11 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-MACOS-ARM64
// RUN: llvm-mc -triple arm64-apple-macos11.1 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-MACOS-ARM64_1
// RUN: llvm-mc -triple arm64-apple-ios13.0-macabi %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-MACCATALYST-ARM64
// RUN: llvm-mc -triple arm64e-apple-ios13.0-macabi %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-MACCATALYST-ARM64
// RUN: llvm-mc -triple arm64-apple-ios14.1-macabi %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-MACCATALYST-ARM64_1

// RUN: llvm-mc -triple arm64e-apple-ios10.3 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOS-ARM64E
// RUN: llvm-mc -triple arm64e-apple-ios13 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOS-ARM64E
// RUN: llvm-mc -triple arm64e-apple-ios14.1 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOS-ARM64E3

// RUN: llvm-mc -triple arm64-apple-ios10-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM2
// RUN: llvm-mc -triple arm64e-apple-ios10-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM2
// RUN: llvm-mc -triple arm64-apple-ios13-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM2
// RUN: llvm-mc -triple arm64-apple-ios14-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM2
// RUN: llvm-mc -triple arm64-apple-ios14.1-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-IOSSIM3
// RUN: llvm-mc -triple arm64-apple-tvos10-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-TVOSSIM2
// RUN: llvm-mc -triple arm64-apple-watchos3-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-BUILD-WATCHOSSIM2

// RUN: llvm-mc -triple arm64-apple-driverkit19.0 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-DRIVERKIT-ARM64
// RUN: llvm-mc -triple arm64e-apple-driverkit19.0 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-DRIVERKIT-ARM64
// RUN: llvm-mc -triple arm64-apple-driverkit20.1 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck %s --check-prefix=CHECK-DRIVERKIT-ARM64_1

// RUN: llvm-mc -triple arm64-apple-xros1 %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck --check-prefix=CHECK-XROS-ARM64 %s
// RUN: llvm-mc -triple arm64-apple-xros1-simulator %s -filetype=obj -o - | llvm-objdump --macho --private-headers - | FileCheck --check-prefix=CHECK-XROS-SIM %s

// CHECK-BUILD-IOS-ARM64E:           cmd LC_BUILD_VERSION
// CHECK-BUILD-IOS-ARM64E-NEXT:  cmdsize 24
// CHECK-BUILD-IOS-ARM64E-NEXT: platform ios
// CHECK-BUILD-IOS-ARM64E-NEXT:      sdk n/a
// CHECK-BUILD-IOS-ARM64E-NEXT:    minos 14.0
// CHECK-BUILD-IOS-ARM64E-NEXT:   ntools 0
// CHECK-BUILD-IOS-ARM64E-NOT: LC_VERSION_MIN

// CHECK-BUILD-IOS-ARM64E3:           cmd LC_BUILD_VERSION
// CHECK-BUILD-IOS-ARM64E3-NEXT:  cmdsize 24
// CHECK-BUILD-IOS-ARM64E3-NEXT: platform ios
// CHECK-BUILD-IOS-ARM64E3-NEXT:      sdk n/a
// CHECK-BUILD-IOS-ARM64E3-NEXT:    minos 14.1
// CHECK-BUILD-IOS-ARM64E3-NEXT:   ntools 0
// CHECK-BUILD-IOS-ARM64E3-NOT: LC_VERSION_MIN

// CHECK-XROS-ARM64:            cmd LC_BUILD_VERSION
// CHECK-XROS-ARM64-NEXT:   cmdsize 24
// CHECK-XROS-ARM64-NEXT:  platform xros
// CHECK-XROS-ARM64-NEXT:       sdk n/a
// CHECK-XROS-ARM64-NEXT:     minos 1.0
// CHECK-XROS-ARM64-NEXT:    ntools 0
// CHECK-XROS-ARM64-NOT: LC_VERSION_MIN

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

// CHECK-XROS-SIM:            cmd LC_BUILD_VERSION
// CHECK-XROS-SIM-NEXT:   cmdsize 24
// CHECK-XROS-SIM-NEXT:  platform xrsimulator
// CHECK-XROS-SIM-NEXT:       sdk n/a
// CHECK-XROS-SIM-NEXT:     minos 1.0
// CHECK-XROS-SIM-NEXT:    ntools 0
// CHECK-XROS-SIM-NOT: LC_VERSION_MIN

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

// CHECK-DRIVERKIT-ARM64:            cmd LC_BUILD_VERSION
// CHECK-DRIVERKIT-ARM64-NEXT:   cmdsize 24
// CHECK-DRIVERKIT-ARM64-NEXT:  platform driverkit
// CHECK-DRIVERKIT-ARM64-NEXT:       sdk n/a
// CHECK-DRIVERKIT-ARM64-NEXT:     minos 20.0
// CHECK-DRIVERKIT-ARM64-NEXT:    ntools 0

// CHECK-DRIVERKIT-ARM64_1:            cmd LC_BUILD_VERSION
// CHECK-DRIVERKIT-ARM64_1-NEXT:   cmdsize 24
// CHECK-DRIVERKIT-ARM64_1-NEXT:  platform driverkit
// CHECK-DRIVERKIT-ARM64_1-NEXT:       sdk n/a
// CHECK-DRIVERKIT-ARM64_1-NEXT:     minos 20.1
// CHECK-DRIVERKIT-ARM64_1-NEXT:    ntools 0
