// RUN: mlir-opt --split-input-file --verify-diagnostics %s 2> %t &&  FileCheck --input-file %t %s
// RUN: cat %t

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
