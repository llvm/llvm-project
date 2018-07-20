// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A0 {
// CHECK:   code Code = [{Simple}];
// CHECK: }

// CHECK: def B0 {
// CHECK:   code Code = [{With paste 7}];
// CHECK: }

class A<code c> {
  code Code = c;
}

def A0 : A<"Simple">;

class B<int i> : A<"With paste " # i>;

def B0 : B<7>;
