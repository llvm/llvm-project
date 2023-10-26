// RUN: mlir-translate -mlir-to-llvmir %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: target datalayout
// CHECK: E-
// CHECK: A4-
// CHECK: S128-
// CHECK: i64:64:128
// CHECK: f80:128:256
// CHECK: p0:32:64:128
// CHECK: p1:32:32:32:16
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<"dlti.endianness", "big">,
#dlti.dl_entry<"dlti.alloca_memory_space", 4 : ui32>,
#dlti.dl_entry<"dlti.stack_alignment", 128 : i32>,
#dlti.dl_entry<index, 64>,
#dlti.dl_entry<i64, dense<[64,128]> : vector<2xi32>>,
#dlti.dl_entry<f80, dense<[128,256]> : vector<2xi32>>,
#dlti.dl_entry<!llvm.ptr, dense<[32,64,128]> : vector<3xi32>>,
#dlti.dl_entry<!llvm.ptr<1>, dense<[32,32,32,16]> : vector<4xi32>>
>} {
  llvm.func @foo() {
    llvm.return
  }
}

// -----

// CHECK: target datalayout
// CHECK: e
// CHECK-NOT: A0
// CHECK-NOT: S0
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<"dlti.endianness", "little">,
#dlti.dl_entry<"dlti.alloca_memory_space", 0 : ui32>,
#dlti.dl_entry<"dlti.stack_alignment", 0 : i32>
>} {
  llvm.func @bar() {
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
