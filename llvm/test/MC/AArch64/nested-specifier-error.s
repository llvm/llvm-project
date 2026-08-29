// REQUIRES: asserts
// RUN: not --crash llvm-mc -triple arm64e-apple-macosx -filetype obj %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: cannot apply another specifier to MCSpecifierExpr
  ldr q4, [x8, :lo12:sym@PAGEOFF]
  add x8, x8, :lo12:sym@PAGEOFF
  add x8, x8, :lo12:sym@PAGEOFF + 4
