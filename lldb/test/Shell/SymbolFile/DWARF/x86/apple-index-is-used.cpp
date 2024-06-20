// Test that we use the apple indexes.
// RUN: %clang %s -g -c -o %t --target=x86_64-apple-macosx -gdwarf-4
// RUN: lldb-test symbols %t | FileCheck %s

// CHECK: .apple_names index present
// CHECK: .apple_types index present

int foo;
