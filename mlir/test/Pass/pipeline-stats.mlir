// RUN: mlir-opt %s -verify-each=true -pass-pipeline='func(test-stats-pass,test-stats-pass)' -pass-statistics -pass-statistics-display=list 2>&1 | FileCheck -check-prefix=LIST %s
// RUN: mlir-opt %s -verify-each=true -pass-pipeline='func(test-stats-pass,test-stats-pass)' -pass-statistics -pass-statistics-display=pipeline 2>&1 | FileCheck -check-prefix=PIPELINE %s

// LIST: Pass statistics report
// LIST: TestStatisticPass
// LIST-NEXT:  (S) {{0|8}} num-ops - Number of operations counted
// LIST-NOT: Verifier

// PIPELINE: Pass statistics report
// PIPELINE: 'func' Pipeline
// PIPELINE-NEXT:   TestStatisticPass
// PIPELINE-NEXT:     (S) {{0|4}} num-ops - Number of operations counted
// PIPELINE-NEXT:   TestStatisticPass
// PIPELINE-NEXT:     (S) {{0|4}} num-ops - Number of operations counted

func @foo() {
  return
}

func @bar() {
  return
}
