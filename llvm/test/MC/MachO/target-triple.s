// REQUIRES: aarch64-registered-target

// RUN: split-file %s %t

// RUN: llvm-mc -triple arm64-apple-macos %t/target-triple.s | FileCheck %t/target-triple.s
// RUN: llvm-mc -triple arm64-apple-macos %t/target-triple.s -filetype=obj -o %t.o
// RUN: llvm-objdump --macho --private-headers %t.o | FileCheck %t/target-triple.s --check-prefix=OBJDUMP

// RUN: not llvm-mc -triple arm64-apple-macos %t/non-darwin-target-triple.s 2>&1 | FileCheck %t/non-darwin-target-triple.s

//--- target-triple.s
.target_triple "arm64-apple-macos27.0.0"
.build_version macos, 27,0 sdk_version 27,0

// CHECK: .target_triple "arm64-apple-macos27.0.0"
// Verify that .target_triple prints a newline so the next statement prints on
// the next line and doesn't run into the .target_triple directive
// CHECK-NEXT: .build_version macos, 27, 0 sdk_version 27, 0


// Make sure the command  is included in the Mach header
// Mach header
// ... filetype ncmds sizeofcmds
// ...   OBJECT     5        320
// (filetype isn't really important for this test, we just didn't want to
// accidentally match a random 5.)
// OBJDUMP: Mach header
// OBJDUMP-NEXT: ncmds
// OBJDUMP-NEXT: OBJECT
// OBJDUMP-SAME: 5
// OBJDUMP-SAME: 320

// OBJDUMP:           cmd LC_TARGET_TRIPLE
// OBJDUMP-NEXT:  cmdsize 40
// OBJDUMP-NEXT:   triple arm64-apple-macos27.0.0

//--- non-darwin-target-triple.s
.target_triple "x86_64-unknown-linux-gnu"

// .target_triple is a Darwin specific directive
// CHECK: error: non-Darwin target triple
