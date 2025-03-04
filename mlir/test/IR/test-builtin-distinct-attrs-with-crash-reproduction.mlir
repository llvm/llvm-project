// This test verifies that when running with crash reproduction enabled, distinct
// attribute storage is not allocated in thread-local storage. Since crash
// reproduction runs the pass manager in a separate thread, using thread-local
// storage for distinct attributes causes use-after-free errors once the thread
// that runs the pass manager joins.

// RUN: mlir-opt --mlir-disable-threading --mlir-pass-pipeline-crash-reproducer=. %s -test-distinct-attrs | FileCheck %s
// RUN: mlir-opt --mlir-pass-pipeline-crash-reproducer=. %s -test-distinct-attrs | FileCheck %s

// CHECK: #[[DIST0:.*]] = distinct[0]<42 : i32>
// CHECK: #[[DIST1:.*]] = distinct[1]<42 : i32>
#distinct = distinct[0]<42 : i32>

// CHECK: @foo_1
func.func @foo_1() {
  // CHECK: "test.op"() {distinct.input = #[[DIST0]], distinct.output = #[[DIST1]]}
  "test.op"() {distinct.input = #distinct} : () -> ()
}
