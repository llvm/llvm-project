// Check near-miss mechanics:
// RUN: mlir-opt --split-input-file --verify-diagnostics %s 2> %t \
// RUN: && FileCheck --input-file %t --check-prefix=CHECK-DEFAULT %s
// RUN: cat %t

// Check that (1) custom input splitter and (2) custom output splitters work.
// RUN: mlir-opt %s -split-input-file="// CHECK-DEFAULT: ""----" \
// RUN:   -output-split-marker="// ---- next split ----" \
// RUN: | FileCheck --check-prefix=CHECK-CUSTOM %s

// Check that (3) the input is not split if `-split-input-file` is not given.
// RUN: mlir-opt %s 2> %t \
// RUN: || FileCheck --input-file %t --check-prefix=CHECK-NOSPLIT %s
// RUN: cat %t

func.func @main() {return}

// -----

// expected-note @+1 {{see existing symbol definition here}}
func.func @foo() { return }
// CHECK-DEFAULT: warning: near miss with file split marker
// CHECK-DEFAULT: ----
// ----

// CHECK-NOSPLIT: error: redefinition of symbol named 'main'
func.func @main() {return}

// expected-error @+1 {{redefinition of symbol named 'foo'}}
func.func @foo() { return }
// CHECK-DEFAULT: warning: near miss with file split marker
// CHECK-DEFAULT: ----
// ----
func.func @bar2() {return }

// No error flagged at the end for a near miss.
// ----

// CHECK-CUSTOM: module
// CHECK-CUSTOM: ---- next split ----
// CHECK-CUSTOM: module
// CHECK-CUSTOM: ---- next split ----
// CHECK-CUSTOM: module
