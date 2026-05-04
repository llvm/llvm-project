// Test that attempting to create a local crash reproducer without disabling threading
// prints an error from the pass manager (as opposed to crashing with a stack trace).

// RUN: mlir-opt --verify-diagnostics --mlir-pass-pipeline-local-reproducer \
// RUN:          --mlir-pass-pipeline-crash-reproducer=%t %s

// expected-error@unknown {{Local crash reproduction may not be used without disabling mutli-threading first.}}
