// Tests that llvm-bcanalyzer recognizes the correct "stream type" for various
// common bitstream formats.

// RUN: llvm-bcanalyzer -dump %s.ast | FileCheck %s -check-prefix=CHECK-AST
// CHECK-AST: Stream type: Clang Serialized AST

// RUN: llvm-bcanalyzer -dump %s.dia | FileCheck %s -check-prefix=CHECK-DIAG
// CHECK-DIAG: Stream type: Clang Serialized Diagnostics

// RUN: not llvm-bcanalyzer -dump %s.ast.incomplete 2>&1 | FileCheck %s -check-prefix=CHECK-INCOMPLETE
// RUN: not llvm-bcanalyzer -dump %s.dia.incomplete 2>&1 | FileCheck %s -check-prefix=CHECK-INCOMPLETE
// CHECK-INCOMPLETE: Bitcode stream should be a multiple of 4 bytes in length

// RUN: llvm-bcanalyzer -dump %s.opt.bitstream | FileCheck %s -check-prefix=CHECK-REMARKS
// CHECK-REMARKS: Stream type: LLVM Remarks
