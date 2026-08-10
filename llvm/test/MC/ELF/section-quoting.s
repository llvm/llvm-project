// RUN: llvm-mc -triple x86_64-pc-linux-gnu %s -o - | FileCheck %s

// Test that we handle the strings like gas
.section bar-"foo"
.section "foo"
.section "foo bar"

// CHECK: .section "bar-\"foo\""
// CHECK: .section foo
// CHECK: .section "foo bar"
