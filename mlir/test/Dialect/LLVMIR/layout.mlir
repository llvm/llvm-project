// RUN: mlir-opt --test-data-layout-query --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  // CHECK: @no_spec
  func @no_spec() {
    // CHECK: alignment = 8
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    "test.data_layout_query"() : () -> !llvm.ptr<i8>
    // CHECK: alignment = 8
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    "test.data_layout_query"() : () -> !llvm.ptr<i32>
    // CHECK: alignment = 8
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    "test.data_layout_query"() : () -> !llvm.ptr<bf16>
    // CHECK: alignment = 8
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    "test.data_layout_query"() : () -> !llvm.ptr<!llvm.ptr<i8>>
    // CHECK: alignment = 8
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    "test.data_layout_query"() : () -> !llvm.ptr<i8, 3>
    // CHECK: alignment = 8
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    "test.data_layout_query"() : () -> !llvm.ptr<i8, 5>
    return
  }
}

// -----

module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr<i8>, dense<[32, 32, 64]> : vector<3xi32>>,
  #dlti.dl_entry<!llvm.ptr<i8, 5>, dense<[64, 64, 64]> : vector<3xi32>>
>} {
  // CHECK: @spec
  func @spec() {
    // CHECK: alignment = 4
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    "test.data_layout_query"() : () -> !llvm.ptr<i8>
    // CHECK: alignment = 4
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    "test.data_layout_query"() : () -> !llvm.ptr<i32>
    // CHECK: alignment = 4
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    "test.data_layout_query"() : () -> !llvm.ptr<bf16>
    // CHECK: alignment = 4
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    "test.data_layout_query"() : () -> !llvm.ptr<!llvm.ptr<i8>>
    // CHECK: alignment = 4
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    "test.data_layout_query"() : () -> !llvm.ptr<i8, 3>
    // CHECK: alignment = 8
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    "test.data_layout_query"() : () -> !llvm.ptr<i8, 5>
    return
  }
}

// -----

// expected-error@below {{unexpected layout attribute for pointer to 'i32'}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr<i32>, dense<[64, 64, 64]> : vector<3xi32>>
>} {
  func @pointer() {
    return
  }
}

// -----

// expected-error@below {{expected layout attribute for '!llvm.ptr<i8>' to be a dense integer elements attribute with 3 or 4 elements}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr<i8>, dense<[64.0, 64.0, 64.0]> : vector<3xf32>>
>} {
  func @pointer() {
    return
  }
}

// -----

// expected-error@below {{preferred alignment is expected to be at least as large as ABI alignment}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr<i8>, dense<[64, 64, 32]> : vector<3xi32>>
>} {
  func @pointer() {
    return
  }
}
