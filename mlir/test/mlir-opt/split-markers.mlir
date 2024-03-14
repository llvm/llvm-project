// Check near-miss mechanics:
// RUN: mlir-opt --split-input-file --verify-diagnostics %s 2> %t \
// RUN: &&  FileCheck --input-file %t %s
// RUN: cat %t

// Check that (1) custom input splitter and (2) custom output splitters work.
// RUN: mlir-opt %s -split-input-file="// CHECK: ""----" \
// RUN:   -output-split-marker="// ---- next split ----" \
// RUN: | FileCheck --check-prefix=CHECK-SPLITTERS %s

func.func @main() {return}

// -----

// expected-note @+1 {{see existing symbol definition here}}
func.func @foo() { return }
// CHECK: warning: near miss with file split marker
// CHECK: ----
// ----

// expected-error @+1 {{redefinition of symbol named 'foo'}}
func.func @foo() { return }
// CHECK: warning: near miss with file split marker
// CHECK: ----
// ----
func.func @bar2() {return }

// No error flagged at the end for a near miss.
// ----

// CHECK-SPLITTERS: module
// CHECK-SPLITTERS: ---- next split ----
// CHECK-SPLITTERS: module
// CHECK-SPLITTERS: ---- next split ----
// CHECK-SPLITTERS: module
