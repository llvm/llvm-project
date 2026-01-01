// RUN: not mlir-opt -allow-unregistered-dialect %s 2>&1 | FileCheck %s

// Test that '//' in a string literal doesn't confuse comment detection when
// backing up to position an error. The error should point to the end of the
// line (column 54), not at the '//' inside the string (column 28).

// CHECK: {{.*}}:[[# @LINE + 2]]:54: error: expected operation name in quotes
func.func @string_with_slashes() {
  "test.op"() {url = "http://example.com"} : () -> ()

