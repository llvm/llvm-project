// RUN: mlir-opt --no-implicit-module --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-NOT: module
//     CHECK: func.func
func.func private @foo()

// -----

// expected-error@-3 {{source must contain a single top-level operation, found: 2}}
func.func private @bar()
func.func private @baz()

// -----

// expected-error@-3 {{source must contain a single top-level operation, found: 0}}
