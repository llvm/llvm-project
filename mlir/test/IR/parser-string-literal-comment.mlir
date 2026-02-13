// RUN: mlir-opt -allow-unregistered-dialect --verify-diagnostics %s

// Test that '//' in a string literal doesn't confuse comment detection when
// backing up to position an error. The error should point to the end of the
// line, not at the '//' inside the string.

func.func @string_with_slashes() {
  // expected-error@+1 {{expected operation name in quotes}}
  "test.op"() {url = "http://example.com"} : () -> ()

