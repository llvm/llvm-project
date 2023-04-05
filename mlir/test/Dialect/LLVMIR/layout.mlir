// RUN: mlir-opt --test-data-layout-query --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  // CHECK: @no_spec
  func.func @no_spec() {
    // CHECK: alignment = 8
    // CHECK: alloca_memory_space = 0
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    // CHECK: stack_alignment = 0
    "test.data_layout_query"() : () -> !llvm.ptr
    // CHECK: alignment = 8
    // CHECK: alloca_memory_space = 0
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    // CHECK: stack_alignment = 0
    "test.data_layout_query"() : () -> !llvm.ptr<3>
    // CHECK: alignment = 8
    // CHECK: alloca_memory_space = 0
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    // CHECK: stack_alignment = 0
    "test.data_layout_query"() : () -> !llvm.ptr<5>
    return
  }
}

// -----

module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr, dense<[32, 32, 64]> : vector<3xi32>>,
  #dlti.dl_entry<!llvm.ptr<5>, dense<[64, 64, 64]> : vector<3xi32>>,
  #dlti.dl_entry<!llvm.ptr<4>, dense<[32, 64, 64]> : vector<3xi32>>,
  #dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>,
  #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>
>} {
  // CHECK: @spec
  func.func @spec() {
    // CHECK: alignment = 4
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    // CHECK: stack_alignment = 128
    "test.data_layout_query"() : () -> !llvm.ptr
    // CHECK: alignment = 4
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    // CHECK: stack_alignment = 128
    "test.data_layout_query"() : () -> !llvm.ptr<3>
    // CHECK: alignment = 8
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 64
    // CHECK: preferred = 8
    // CHECK: size = 8
    // CHECK: stack_alignment = 128
    "test.data_layout_query"() : () -> !llvm.ptr<5>
    // CHECK: alignment = 4
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    // CHECK: stack_alignment = 128
    "test.data_layout_query"() : () -> !llvm.ptr<3>
    // CHECK: alignment = 8
    // CHECK: alloca_memory_space = 5
    // CHECK: bitsize = 32
    // CHECK: preferred = 8
    // CHECK: size = 4
    // CHECK: stack_alignment = 128
	  "test.data_layout_query"() : () -> !llvm.ptr<4>
    return
  }
}

// -----

// expected-error@below {{expected layout attribute for '!llvm.ptr' to be a dense integer elements attribute with 3 or 4 elements}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr, dense<[64.0, 64.0, 64.0]> : vector<3xf32>>
>} {
  func.func @pointer() {
    return
  }
}

// -----

// expected-error@below {{preferred alignment is expected to be at least as large as ABI alignment}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr, dense<[64, 64, 32]> : vector<3xi32>>
>} {
  func.func @pointer() {
    return
  }
}

// -----

module {
    // CHECK: @no_spec
    func.func @no_spec() {
        // simple case
        // CHECK: alignment = 4
        // CHECK: bitsize = 32
        // CHECK: preferred = 4
        // CHECK: size = 4
        "test.data_layout_query"() : () -> !llvm.struct<(i32)>

        // padding inbetween
        // CHECK: alignment = 8
        // CHECK: bitsize = 128
        // CHECK: preferred = 8
        // CHECK: size = 16
        "test.data_layout_query"() : () -> !llvm.struct<(i32, f64)>

        // padding at end of struct
        // CHECK: alignment = 8
        // CHECK: bitsize = 128
        // CHECK: preferred = 8
        // CHECK: size = 16
        "test.data_layout_query"() : () -> !llvm.struct<(f64, i32)>

         // packed
         // CHECK: alignment = 1
         // CHECK: bitsize = 96
         // CHECK: preferred = 8
         // CHECK: size = 12
         "test.data_layout_query"() : () -> !llvm.struct<packed (f64, i32)>

         // empty
         // CHECK: alignment = 1
         // CHECK: bitsize = 0
         // CHECK: preferred = 1
         // CHECK: size = 0
         "test.data_layout_query"() : () -> !llvm.struct<()>
         return
    }
}

// -----

module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.struct<()>, dense<[32, 32]> : vector<2xi32>>
>} {
    // CHECK: @spec
    func.func @spec() {
        // Strict alignment is applied
        // CHECK: alignment = 4
        // CHECK: bitsize = 16
        // CHECK: preferred = 4
        // CHECK: size = 2
        "test.data_layout_query"() : () -> !llvm.struct<(i16)>

        // No impact on structs that have stricter requirements
        // CHECK: alignment = 8
        // CHECK: bitsize = 128
        // CHECK: preferred = 8
        // CHECK: size = 16
        "test.data_layout_query"() : () -> !llvm.struct<(i32, f64)>

         // Only the preferred alignment of structs is affected
         // CHECK: alignment = 1
         // CHECK: bitsize = 32
         // CHECK: preferred = 4
         // CHECK: size = 4
         "test.data_layout_query"() : () -> !llvm.struct<packed (i16, i16)>

         // empty
         // CHECK: alignment = 4
         // CHECK: bitsize = 0
         // CHECK: preferred = 4
         // CHECK: size = 0
         "test.data_layout_query"() : () -> !llvm.struct<()>
         return
    }
}

// -----

module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.struct<()>, dense<[32]> : vector<1xi32>>
>} {
    // CHECK: @spec_without_preferred
    func.func @spec_without_preferred() {
        // abi alignment is applied to both preferred and abi
        // CHECK: alignment = 4
        // CHECK: bitsize = 16
        // CHECK: preferred = 4
        // CHECK: size = 2
        "test.data_layout_query"() : () -> !llvm.struct<(i16)>
        return
    }
}

// -----

// expected-error@below {{unexpected layout attribute for struct '!llvm.struct<(i8)>'}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.struct<(i8)>, dense<[64, 64]> : vector<2xi32>>
>} {
  func.func @struct() {
    return
  }
}

// -----

// expected-error@below {{expected layout attribute for '!llvm.struct<()>' to be a dense integer elements attribute of 1 or 2 elements}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.struct<()>, dense<[64, 64, 64]> : vector<3xi32>>
>} {
  func.func @struct() {
    return
  }
}

// -----

// expected-error@below {{preferred alignment is expected to be at least as large as ABI alignment}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.struct<()>, dense<[64, 32]> : vector<2xi32>>
>} {
  func.func @struct() {
    return
  }
}

// -----

module {
    // CHECK: @arrays
    func.func @arrays() {
        // simple case
        // CHECK: alignment = 4
        // CHECK: bitsize = 64
        // CHECK: preferred = 4
        // CHECK: size = 8
        "test.data_layout_query"() : () -> !llvm.array<2 x i32>

        // size 0
        // CHECK: alignment = 8
        // CHECK: bitsize = 0
        // CHECK: preferred = 8
        // CHECK: size = 0
        "test.data_layout_query"() : () -> !llvm.array<0 x f64>

        // alignment info matches element type
        // CHECK: alignment = 4
        // CHECK: bitsize = 64
        // CHECK: preferred = 8
        // CHECK: size = 8
        "test.data_layout_query"() : () -> !llvm.array<1 x i64>
        return
    }
}

// -----

module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.struct<()>, dense<[64]> : vector<1xi32>>
>} {
    // CHECK: @overaligned
    func.func @overaligned() {
        // Over aligned element types are respected
        // CHECK: alignment = 8
        // CHECK: bitsize = 128
        // CHECK: preferred = 8
        // CHECK: size = 16
        "test.data_layout_query"() : () -> !llvm.array<2 x struct<(i8)>>
         return
    }
}
