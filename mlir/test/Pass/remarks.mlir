// RUN: mlir-opt %s --test-remark --remarks-filter-passed="category-1-passed" 2>&1 | FileCheck %s -check-prefix=CHECK-PASSED 
// RUN: mlir-opt %s --test-remark --remarks-filter-missed="a-category-1-missed" 2>&1 | FileCheck %s -check-prefix=CHECK-MISSED
// RUN: mlir-opt %s --test-remark --remarks-filter-failed="category-2-failed" 2>&1 | FileCheck %s -check-prefix=CHECK-FAILED
// RUN: mlir-opt %s --test-remark --remarks-filter-analyse="category-2-analysis" 2>&1 | FileCheck %s -check-prefix=CHECK-ANALYSIS
// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" 2>&1 | FileCheck %s -check-prefix=CHECK-ALL
// RUN: mlir-opt %s --test-remark --remarks-filter="category-1.*" 2>&1 | FileCheck %s -check-prefix=CHECK-ALL1
module @foo {
  "test.op"() : () -> ()
  
}


// CHECK-PASSED: remarks.mlir:8:3: remark: [Passed] test-remark | Category:category-1-passed | Reason="because we are testing the remark pipeline", Remark="This is a test passed remark", Suggestion="try using the remark pipeline feature"
// CHECK-MISSED:remarks.mlir:8:3: remark: [Missed] test-remark | Category:a-category-1-missed | Reason="because we are testing the remark pipeline", Remark="This is a test missed remark", Suggestion="try using the remark pipeline feature"
// CHECK-FAILED: remarks.mlir:8:3: remark: [Failure] test-remark | Category:category-2-failed | Reason="because we are testing the remark pipeline", Remark="This is a test failed remark", Suggestion="try using the remark pipeline feature"
// CHECK-ANALYSIS: remarks.mlir:8:3: remark: [Analysis] test-remark | Category:category-2-analysis | Remark="This is a test analysis remark"


// CHECK-ALL: remarks.mlir:8:3: remark: [Passed] test-remark | Category:category-1-passed | Reason="because we are testing the remark pipeline", Remark="This is a test passed remark", Suggestion="try using the remark pipeline feature"
// CHECK-ALL: remarks.mlir:8:3: remark: [Failure] test-remark | Category:category-2-failed | Reason="because we are testing the remark pipeline", Remark="This is a test failed remark", Suggestion="try using the remark pipeline feature"
// CHECK-ALL: remarks.mlir:8:3: remark: [Analysis] test-remark | Category:category-2-analysis | Remark="This is a test analysis remark"

// CHECK-ALL1: remarks.mlir:8:3: remark: [Passed] test-remark | Category:category-1-passed | Reason="because we are testing the remark pipeline", Remark="This is a test passed remark", Suggestion="try using the remark pipeline feature"
// CHECK-ALL1-NOT: remarks.mlir:8:3: remark: [Missed]
// CHECK-ALL1-NOT: remarks.mlir:8:3: remark: [Failure]
// CHECK-ALL1-NOT: remarks.mlir:8:3: remark: [Analysis]


