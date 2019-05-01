// RUN: llvm-mc -triple aarch64-apple-ios -o - -filetype=obj %s | \
// RUN:   llvm-readobj --symbols | FileCheck %s -check-prefix=READOBJ

// READOBJ-LABEL: Name: cold_func
// READOBJ-NEXT: Type: Section
// READOBJ-NEXT: Section: __text
// READOBJ-NEXT: RefType: UndefinedNonLazy (0x0)
// READOBJ-NEXT: Flags [ (0x400)

  .text
  .cold cold_func
cold_func:
  ret
