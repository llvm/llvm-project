// RUN: not llvm-mc -g -triple i386-unknown-unknown %s 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err
// Test for Bug 11740

        .file "hello"
        .file 1 "world"

// CHECK: .file "hello"
// CHECK-ERRORS:6:9: error: input can't have .file dwarf directives when -g is used to generate dwarf debug info for assembly code
