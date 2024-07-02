// Check that (1) the output split marker is inserted and (2) the input file is
// split using the default split marker.
// RUN: mlir-translate %s -split-input-file -mlir-to-llvmir \
// RUN:   -output-split-marker="; -----" \
// RUN: | FileCheck -check-prefix=CHECK-OUTPUT %s

// With the second command, check that (3) the input split marker is used and
// (4) the output split marker is empty if not specified.
// RUN: mlir-translate %s -split-input-file="// ""-----" -mlir-to-llvmir \
// RUN:   -output-split-marker="; -----" \
// RUN: | mlir-translate -split-input-file -import-llvm \
// RUN:   -split-input-file="; -----" \
// RUN: | FileCheck -check-prefix=CHECK-ROUNDTRIP %s

// Check that (5) the input is not split if `-split-input-file` is not given.
// RUN: mlir-translate %s -mlir-to-llvmir \
// RUN: | FileCheck -check-prefix=CHECK-NOSPLIT %s

// CHECK-OUTPUT:      ModuleID
// CHECK-OUTPUT:      ; -----
// CHECK-OUTPUT-NEXT: ModuleID

// CHECK-ROUNDTRIP:       module {{.*}} {
// CHECK-ROUNDTRIP-NEXT:  }
// CHECK-ROUNDTRIP-EMPTY:
// CHECK-ROUNDTRIP:       module

// CHECK-NOSPLIT:     ModuleID
// CHECK-NOSPLIT-NOT: ModuleID

module {}

// -----

module {}
