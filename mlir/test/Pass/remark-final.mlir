// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final --remark-format=yaml --remarks-output-file=%t.yaml
// RUN: FileCheck %s < %t.yaml
module @foo {
  "test.op"() : () -> ()
  
}

// CHECK-NOT: This is a test passed remark (should be dropped)

// CHECK-DAG: !Analysis
// CHECK-DAG: !Failure
// CHECK-DAG: !Passed
