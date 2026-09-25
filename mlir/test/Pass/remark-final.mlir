// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final 2>&1 | FileCheck %s --implicit-check-not="remark:" --implicit-check-not="should be dropped"
// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final --remark-format=yaml --remarks-output-file=%t.yaml
// RUN: FileCheck --check-prefix=CHECK-YAML %s --implicit-check-not="--- !" --implicit-check-not="should be dropped" < %t.yaml
module @foo {
  "test.op"() : () -> ()
}

// mlir-opt calls finalize() explicitly and the engine destructor calls it
// again; the second call must not emit the remarks a second time.
// --implicit-check-not pins the number of "remark:" lines and of YAML records
// to five.

// CHECK-DAG: remark: [Passed] test-remark | Category:category-1-passed |{{.*}}Remark="This is a test passed remark",
// CHECK-DAG: remark: [Failure] test-remark | Category:category-2-failed
// CHECK-DAG: remark: [Analysis] test-remark | Category:category-2-analysis
// CHECK-DAG: remark: [Passed] test-remark | Category:category-link |{{.*}}RelatedTo=
// CHECK-DAG: remark: [Analysis] test-remark | Category:category-link

// CHECK-YAML-DAG: --- !Passed
// CHECK-YAML-DAG: --- !Failure
// CHECK-YAML-DAG: --- !Analysis
// CHECK-YAML-DAG: --- !Passed
// CHECK-YAML-DAG: --- !Analysis
