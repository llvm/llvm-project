// RUN: not llvm-mc -triple i386 %s -o /dev/null 2>&1 | FileCheck %s

.ifeqs

// CHECK: error: expected string parameter for '.ifeqs' directive
// CHECK: .ifeqs
// CHECK:       ^

.ifeqs "string1"

// CHECK: error: expected comma after first string for '.ifeqs' directive
// CHECK: .ifeqs "string1"
// CHECK:                 ^

.ifeqs "string1",

// CHECK: error: expected string parameter for '.ifeqs' directive
// CHECK: .ifeqs "string1",
// CHECK:                  ^

// CHECK-NOT: error: unmatched .ifs or .elses

.ifnes

// CHECK: error: expected string parameter for '.ifnes' directive
// CHECK: .ifnes
// CHECK:       ^

.ifnes "string1"

// CHECK: error: expected comma after first string for '.ifnes' directive
// CHECK: .ifnes "string1"
// CHECK:                 ^

.ifnes "string1",

// CHECK: error: expected string parameter for '.ifnes' directive
// CHECK: .ifnes "string1",
// CHECK:                  ^
