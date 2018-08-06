// RUN: llvm-mc -g -triple i386-unknown-unknown %s | FileCheck %s
// Test for Bug 11740
// This testcase has two directive files,
// when compiled with -g, this testcase will not report error,
// but keep the debug info existing in the assembly file.

        .file "hello"
        .file 1 "world"

// CHECK: .file "hello"
// CHECK: .file 1 "world"
