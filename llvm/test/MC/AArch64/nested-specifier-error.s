// RUN: not llvm-mc -triple arm64e-apple-macosx -filetype obj %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: [[@LINE+1]]:26: error: invalid variant on expression 'PAGEOFF' (already modified)
  ldr q4, [x8, :lo12:sym@PAGEOFF]

// CHECK: [[@LINE+1]]:25: error: invalid variant on expression 'PAGEOFF' (already modified)
  add x8, x8, :lo12:sym@PAGEOFF

// CHECK: [[@LINE+1]]:25: error: invalid variant on expression 'PAGEOFF' (already modified)
  add x8, x8, :lo12:sym@PAGEOFF + 4
