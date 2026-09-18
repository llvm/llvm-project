// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final 2>&1 | FileCheck %s --implicit-check-not="remark:" --implicit-check-not="should be dropped"
// RUN: mlir-opt %s --test-remark --remarks-filter="category.*" --remark-policy=final --remark-format=yaml --remarks-output-file=%t.yaml
// RUN: FileCheck --check-prefix=CHECK-YAML %s --implicit-check-not="--- !" --implicit-check-not="should be dropped" < %t.yaml
module @foo {
  "test.op"() : () -> ()
}

// The two passed remarks in "category-1-passed" share an identity, so only the
// second survives, in the first one's position. mlir-opt calls finalize()
// explicitly and the engine destructor calls it again. The second call must not
// emit the remarks a second time. --implicit-check-not pins the number of
// "remark:" lines and of YAML records to five.

// CHECK: remark: [Passed] test-remark | Category:category-1-passed |{{.*}}Remark="This is a test passed remark",
// CHECK: remark: [Failure] test-remark | Category:category-2-failed
// CHECK: remark: [Analysis] test-remark | Category:category-2-analysis
// CHECK: remark: [Passed] test-remark | Category:category-link |{{.*}}RelatedTo=
// CHECK: remark: [Analysis] test-remark | Category:category-link

// CHECK-YAML:      --- !Passed
// CHECK-YAML-NEXT: Pass:{{.*}}category-1-passed
// CHECK-YAML:      --- !Failure
// CHECK-YAML-NEXT: Pass:{{.*}}category-2-failed
// CHECK-YAML:      --- !Analysis
// CHECK-YAML-NEXT: Pass:{{.*}}category-2-analysis
// CHECK-YAML:      --- !Passed
// CHECK-YAML-NEXT: Pass:{{.*}}category-link
// CHECK-YAML:      --- !Analysis
// CHECK-YAML-NEXT: Pass:{{.*}}category-link
