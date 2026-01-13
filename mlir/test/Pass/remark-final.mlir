// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final 2>&1 | FileCheck %s 
// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final --remark-format=yaml --remarks-output-file=%t.yaml
// RUN: FileCheck --check-prefix=CHECK-YAML %s < %t.yaml
module @foo {
  "test.op"() : () -> ()
  
}

// CHECK-YAML-NOT: This is a test passed remark (should be dropped)
// CHECK-YAML-DAG: !Analysis
// CHECK-YAML-DAG: !Failure
// CHECK-YAML-DAG: !Passed

// CHECK-NOT: This is a test passed remark (should be dropped)
// CHECK-DAG: remark: [Analysis] test-remark
// CHECK-DAG: remark: [Failure] test-remark | Category:category-2-failed
// CHECK-DAG: remark: [Passed] test-remark | Category:category-1-passed
