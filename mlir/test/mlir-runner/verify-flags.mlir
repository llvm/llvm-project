// REQUIRES: asserts
// RUN: mlir-runner %s --debug-only=jit-runner -mattr=+foo_bar -e entry -entry-point-result=void 2>&1 | FileCheck %s --check-prefixes=MATTR
// RUN: not mlir-runner %s --debug-only=jit-runner -march=bar_foo -e entry -entry-point-result=void 2>&1 | FileCheck %s --check-prefixes=MARCH

// Verify that command line args do affect the configuration

// MATTR: Features = 
// MATTR-SAME: +foo_bar

// MARCH: Failed to create a TargetMachine for the host
// MARCH-NEXT: No available targets are compatible with triple "bar_foo-{{.*}}"

llvm.func @entry() -> () {
  llvm.return
}
