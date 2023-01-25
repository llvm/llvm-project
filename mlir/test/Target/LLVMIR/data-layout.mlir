// RUN: mlir-translate -mlir-to-llvmir %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: target datalayout
// CHECK: E-
// CHECK: i64:64:128
// CHECK: f80:128:256
// CHECK: p0:32:64:128
// CHECK: p1:32:32:32:64
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<"dlti.endianness", "big">,
#dlti.dl_entry<index, 64>,
#dlti.dl_entry<i64, dense<[64,128]> : vector<2xi32>>,
#dlti.dl_entry<f80, dense<[128,256]> : vector<2xi32>>,
#dlti.dl_entry<!llvm.ptr, dense<[32,64,128]> : vector<3xi32>>,
#dlti.dl_entry<!llvm.ptr<1>, dense<[32,32,32,64]> : vector<4xi32>>
>} {
  llvm.func @foo() {
    llvm.return
  }
}

// -----

// expected-error@below {{unsupported data layout for non-signless integer 'ui64'}}
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<ui64, dense<[64,128]> : vector<2xi32>>>
} {}

// -----

// expected-error@below {{unsupported type in data layout: 'bf16'}}
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<bf16, dense<[64,128]> : vector<2xi32>>>
} {}

// -----

// expected-error@below {{unsupported data layout key "foo"}}
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<"foo", dense<[64,128]> : vector<2xi32>>>
} {}
