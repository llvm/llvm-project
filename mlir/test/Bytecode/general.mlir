// RUN: mlir-opt -allow-unregistered-dialect -emit-bytecode %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: s390x-

// CHECK-LABEL: "bytecode.test1"
// CHECK-NEXT:    "bytecode.empty"() : () -> ()
// CHECK-NEXT:    "bytecode.attributes"() {attra = 10 : i64, attrb = #bytecode.attr} : () -> ()
// CHECK-NEXT:    test.graph_region {
// CHECK-NEXT:      "bytecode.operands"(%[[RESULTS:.*]]#0, %[[RESULTS]]#1, %[[RESULTS]]#2) : (i32, i64, i32) -> ()
// CHECK-NEXT:      %[[RESULTS]]:3 = "bytecode.results"() : () -> (i32, i64, i32)
// CHECK-NEXT:    }
// CHECK-NEXT:    "bytecode.branch"()[^[[BLOCK:.*]]] : () -> ()
// CHECK-NEXT:  ^[[BLOCK]](%[[ARG0:.*]]: i32, %[[ARG1:.*]]: !bytecode.int, %[[ARG2:.*]]: !pdl.operation):
// CHECK-NEXT:    "bytecode.regions"() ({
// CHECK-NEXT:      "bytecode.operands"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (i32, !bytecode.int, !pdl.operation) -> ()
// CHECK-NEXT:      "bytecode.return"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "bytecode.return"() : () -> ()
// CHECK-NEXT:  }) : () -> ()

"bytecode.test1"() ({
  "bytecode.empty"() : () -> ()
  "bytecode.attributes"() {attra = 10, attrb = #bytecode.attr} : () -> ()
  test.graph_region {
    "bytecode.operands"(%results#0, %results#1, %results#2) : (i32, i64, i32) -> ()
    %results:3 = "bytecode.results"() : () -> (i32, i64, i32)
  }
  "bytecode.branch"()[^secondBlock] : () -> ()

^secondBlock(%arg1: i32, %arg2: !bytecode.int, %arg3: !pdl.operation):
  "bytecode.regions"() ({
    "bytecode.operands"(%arg1, %arg2, %arg3) : (i32, !bytecode.int, !pdl.operation) -> ()
    "bytecode.return"() : () -> ()
  }) : () -> ()
  "bytecode.return"() : () -> ()
}) : () -> ()
