// Test that attempting to create a local crash reproducer without disabling threading
// prints an error from the pass manager (as opposed to crashing with a stack trace).

// We need to use `||` in the RUN command because lit will fail the test due to mlir-opt
// returning non-zero status for this test case, however this is the intended behaviour.

// RUN: mlir-opt --mlir-pass-pipeline-local-reproducer --mlir-pass-pipeline-crash-reproducer=%t %s 2>&1 || FileCheck --input-file %s %s

// CHECK: error: Local crash reproduction may not be used without disabling mutli-threading first.
