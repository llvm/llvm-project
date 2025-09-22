// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final --remark-format=yaml --remarks-output-file=%t.yaml
// RUN: FileCheck %t.yaml < %s 
module @foo {
  "test.op"() : () -> ()
  
}

// CHECK-NOT: This is a test passed remark (should be dropped)

// CHECK: --- !Failure
// CHECK: Name:            test-remark
// CHECK: DebugLoc:        { File: '../mlir/test/Pass/remark-final.mlir', Line: 4, 
// CHECK:                    Column: 3 }
// CHECK: Function:        '<unknown function>'
// CHECK: Args:
// CHECK:   - Remark:          This is a test failed remark
// CHECK:   - Reason:          because we are testing the remark pipeline
// CHECK:   - Suggestion:      try using the remark pipeline feature

// CHECK: --- !Passed
// CHECK: Pass:            category-1-passed
// CHECK: Name:            test-remark
// CHECK: DebugLoc:        { File: '../mlir/test/Pass/remark-final.mlir', Line: 4, 
// CHECK-NEXT:                    Column: 3 }
// CHECK: Function:        '<unknown function>'
// CHECK: Args:
// CHECK:   - Remark:          This is a test passed remark
// CHECK:   - Reason:          because we are testing the remark pipeline
// CHECK:   - Suggestion:      try using the remark pipeline feature

// CHECK: --- !Analysis
// CHECK: Pass:            category-2-analysis
// CHECK: Name:            test-remark
// CHECK: DebugLoc:        { File: '../mlir/test/Pass/remark-final.mlir', Line: 4, 
// CHECK-NEXT:                    Column: 3 }
// CHECK: Function:        '<unknown function>'
// CHECK: Args:
// CHECK:   - Remark:          This is a test analysis remark