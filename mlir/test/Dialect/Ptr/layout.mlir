// RUN: mlir-opt --test-data-layout-query --split-input-file --verify-diagnostics %s | FileCheck %s

module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!ptr.ptr, dense<[32, 32, 64]> : vector<3xi64>>,
  #dlti.dl_entry<!ptr.ptr<5>, dense<[64, 64, 64]> : vector<3xi64>>,
  #dlti.dl_entry<!ptr.ptr<4>, dense<[32, 64, 64, 24]> : vector<4xi64>>,
  #dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui64>,
  #dlti.dl_entry<"dlti.global_memory_space", 2 : ui64>,
  #dlti.dl_entry<"dlti.program_memory_space", 3 : ui64>,
  #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>
>} {
  // CHECK: @spec
  func.func @spec() {
    // CHECK: alignment = 4
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 32
    // CHECK: global_memory_space = 2
    // CHECK: index = 32
    // CHECK: preferred = 8
    // CHECK: program_memory_space = 3
    // CHECK: size = 4
    // CHECK: stack_alignment = 128
    "test.data_layout_query"() : () -> !ptr.ptr
    // CHECK: alignment = 4
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 32
    // CHECK: global_memory_space = 2
    // CHECK: index = 32
    // CHECK: preferred = 8
    // CHECK: program_memory_space = 3
    // CHECK: size = 4
    // CHECK: stack_alignment = 128
    "test.data_layout_query"() : () -> !ptr.ptr<3>
    // CHECK: alignment = 8
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 64
    // CHECK: global_memory_space = 2
    // CHECK: index = 64
    // CHECK: preferred = 8
    // CHECK: program_memory_space = 3
    // CHECK: size = 8
    // CHECK: stack_alignment = 128
    "test.data_layout_query"() : () -> !ptr.ptr<5>
    // CHECK: alignment = 8
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 32
    // CHECK: global_memory_space = 2
    // CHECK: index = 24
    // CHECK: preferred = 8
    // CHECK: program_memory_space = 3
    // CHECK: size = 4
    // CHECK: stack_alignment = 128
    "test.data_layout_query"() : () -> !ptr.ptr<4>
    return
  }
}

// -----

// expected-error@below {{expected layout attribute for '!ptr.ptr' to be a dense integer elements attribute with 3 or 4 elements}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!ptr.ptr, dense<[64.0, 64.0, 64.0]> : vector<3xf32>>
>} {
  func.func @pointer() {
    return
  }
}

// -----

// expected-error@below {{preferred alignment is expected to be at least as large as ABI alignment}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!ptr.ptr, dense<[64, 64, 32]> : vector<3xi64>>
>} {
  func.func @pointer() {
    return
  }
}

// -----

// expected-error @below {{expected i64 parameters for '!ptr.ptr'}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!ptr.ptr, dense<[32, 32, 64]> : vector<3xi32>>
>} {
}

