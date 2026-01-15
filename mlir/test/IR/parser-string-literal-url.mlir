// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

// Test that URLs with '//' in string literals parse correctly.

// CHECK-LABEL: func.func @url_in_string
// CHECK: "test.op"() {url = "http://example.com/path"} : () -> ()
// CHECK: "test.op"() {url = "https://example.com//double//slashes"} : () -> ()
// CHECK: "test.op"() {proto = "file:///local/path"} : () -> ()
func.func @url_in_string() {
  "test.op"() {url = "http://example.com/path"} : () -> ()
  "test.op"() {url = "https://example.com//double//slashes"} : () -> ()
  "test.op"() {proto = "file:///local/path"} : () -> ()
  return
}
