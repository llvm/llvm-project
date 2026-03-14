// REQUIRES: asserts
// Check that statistics in nested pipelines are not ignored and that this works with and without threading.
// RUN: mlir-opt %s -verify-each=true -pass-pipeline='builtin.module(builtin.module(func.func(test-stats-pass,test-stats-pass)))' -mlir-pass-statistics -mlir-pass-statistics-display=list -mlir-disable-threading 2>&1 | FileCheck -check-prefix=LIST %s
// RUN: mlir-opt %s -verify-each=true -pass-pipeline='builtin.module(builtin.module(func.func(test-stats-pass,test-stats-pass)))' -mlir-pass-statistics -mlir-pass-statistics-display=list                         2>&1 | FileCheck -check-prefix=LIST %s

// RUN: mlir-opt %s -verify-each=true -pass-pipeline='builtin.module(builtin.module(func.func(test-stats-pass,test-stats-pass)))' -mlir-pass-statistics -mlir-pass-statistics-display=pipeline -mlir-disable-threading 2>&1 | FileCheck -check-prefix=PIPELINE %s
// RUN: mlir-opt %s -verify-each=true -pass-pipeline='builtin.module(builtin.module(func.func(test-stats-pass,test-stats-pass)))' -mlir-pass-statistics -mlir-pass-statistics-display=pipeline                         2>&1 | FileCheck -check-prefix=PIPELINE %s

// Check for the correct statistic values.
// Each test-stats-pass will count two ops: the func.func and the func.return .
//      LIST: (S) 4 num-ops
// LIST-NEXT: (S) 4 num-ops2
//      PIPELINE: (S) 2 num-ops
// PIPELINE-NEXT: (S) 2 num-ops2
//      PIPELINE: (S) 2 num-ops
// PIPELINE-NEXT: (S) 2 num-ops2

module {
  module {
    func.func @foo() {
      return
    }
  }
}
