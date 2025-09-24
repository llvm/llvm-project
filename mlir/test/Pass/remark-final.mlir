// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final --remark-format=yaml --remarks-output-file=%t.yaml
// RUN: FileCheck %s < %t.yaml
module @foo {
  "test.op"() : () -> ()
  
}

// CHECK-NOT: This is a test passed remark (should be dropped)

// CHECK: !Passed
// CHECK: Remark:          This is a test passed remark
// CHECK: !Analysis
// CHECK: Remark:          This is a test analysis remark
// CHECK: !Failure
// CHECK: Remark:          This is a test failed remark
