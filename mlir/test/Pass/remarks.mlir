// RUN: mlir-opt %s --test-remark --remarks-filter-passed="category-1-passed" 2>&1 | FileCheck %s -check-prefix=CHECK-PASSED 
// RUN: mlir-opt %s --test-remark --remarks-filter-missed="a-category-1-missed" 2>&1 | FileCheck %s -check-prefix=CHECK-MISSED
// RUN: mlir-opt %s --test-remark --remarks-filter-failed="category-2-failed" 2>&1 | FileCheck %s -check-prefix=CHECK-FAILED
// RUN: mlir-opt %s --test-remark --remarks-filter-analyse="category-2-analysis" 2>&1 | FileCheck %s -check-prefix=CHECK-ANALYSIS
// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" 2>&1 | FileCheck %s -check-prefix=CHECK-ALL
// RUN: mlir-opt %s --test-remark --remarks-filter="category-1.*" 2>&1 | FileCheck %s -check-prefix=CHECK-ALL1
// RUN: mlir-opt %s --test-remark --remark-policy=final --remarks-filter-passed="category-link" --remarks-filter-analyse="category-link" 2>&1 | FileCheck %s -check-prefix=CHECK-LINK
module @foo {
  "test.op"() : () -> ()
}

// Each remark now includes a RemarkId=N arg and remark-N prefix, so use
// {{.*}} to absorb them.
// CHECK-PASSED: remark: [Passed]{{.*}}test-remark | Category:category-1-passed |{{.*}}Remark="This is a test passed remark"
// CHECK-MISSED: remark: [Missed]{{.*}}test-remark | Category:a-category-1-missed |{{.*}}Remark="This is a test missed remark"
// CHECK-FAILED: remark: [Failure]{{.*}}test-remark | Category:category-2-failed |{{.*}}Remark="This is a test failed remark"
// CHECK-ANALYSIS: remark: [Analysis]{{.*}}test-remark | Category:category-2-analysis |{{.*}}Remark="This is a test analysis remark"

// CHECK-ALL: remark: [Passed]{{.*}}test-remark | Category:category-1-passed |
// CHECK-ALL: remark: [Passed]{{.*}}test-remark | Category:category-1-passed |
// CHECK-ALL: remark: [Failure]{{.*}}test-remark | Category:category-2-failed |
// CHECK-ALL: remark: [Analysis]{{.*}}test-remark | Category:category-2-analysis |
// CHECK-ALL: remark: [Passed]{{.*}}test-remark | Category:category-link |
// CHECK-ALL: remark: [Analysis]{{.*}}test-remark | Category:category-link |

// CHECK-ALL1: remark: [Passed]{{.*}}test-remark | Category:category-1-passed |
// CHECK-ALL1-NOT: remark: [Missed]
// CHECK-ALL1-NOT: remark: [Failure]
// CHECK-ALL1-NOT: remark: [Analysis]

// Test remark linking via query-based API with PolicyFinal: the passed remark
// finds the analysis remark by searching postponedRemarks. Related (child)
// remarks are grouped after their parent.
// CHECK-LINK: remark: [Passed]{{.*}}Category:category-link |{{.*}}RelatedTo=
// CHECK-LINK-SAME: Remark="Optimization enabled by analysis"
// CHECK-LINK-SAME: RemarkId=
// CHECK-LINK: remark: [Analysis]{{.*}}Category:category-link |{{.*}}Remark="Analysis that enables optimization"
// CHECK-LINK-SAME: RemarkId=
