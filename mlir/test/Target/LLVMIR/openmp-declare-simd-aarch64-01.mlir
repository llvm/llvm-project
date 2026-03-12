// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

module attributes {
  llvm.target_triple = "aarch64-unknown-linux-gnu",
  llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
} {
  llvm.func @foo(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    omp.declare_simd simdlen(2)
    omp.declare_simd simdlen(6)
    omp.declare_simd simdlen(8)
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }

  // Make sure that the following two functions by default get generated
  // with 4 and 2 lanes, as described in the vector ABI.

  llvm.func @bar(%x: f64) -> f32 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd notinbranch
    %0 = llvm.fptrunc %x : f64 to f32
    llvm.return %0 : f32
  }

  llvm.func @baz(%x: f32) -> f64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd notinbranch
    %0 = llvm.fpext %x : f32 to f64
    llvm.return %0 : f64
  }

  llvm.func @foo_int(%x: i32) -> i64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    omp.declare_simd simdlen(2)
    omp.declare_simd simdlen(6)
    omp.declare_simd simdlen(8)
    %0 = llvm.sext %x : i32 to i64
    llvm.return %0 : i64
  }

  llvm.func @simple_8bit(%x: i8) -> i8 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    llvm.return %x : i8
  }

  llvm.func @simple_16bit(%x: i16) -> i16 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    llvm.return %x : i16
  }

  llvm.func @simple_32bit(%x: i32) -> i32 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    llvm.return %x : i32
  }

  llvm.func @simple_64bit(%x: i64) -> i64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    llvm.return %x : i64
  }

  llvm.func @a01(%x: i32) -> i8 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    omp.declare_simd simdlen(32)
    %0 = llvm.trunc %x : i32 to i8
    llvm.return %0 : i8
  }

  llvm.func @a02(%x: i16) -> i64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    omp.declare_simd simdlen(2)
    %0 = llvm.sext %x : i16 to i64
    llvm.return %0 : i64
  }

  // ************
  // * pointers *
  // ************

  llvm.func @b01(%x: !llvm.ptr) -> i32 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }

  llvm.func @b02(%x: !llvm.ptr) -> i8 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    %0 = llvm.mlir.constant(0 : i8) : i8
    llvm.return %0 : i8
  }

  llvm.func @b03(%x: !llvm.ptr) -> !llvm.ptr attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd
    llvm.return %x : !llvm.ptr
  }

  // ***********
  // * masking *
  // ***********

  llvm.func @c01(%x: !llvm.ptr, %y: i16) -> i32 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd inbranch
    %0 = llvm.sext %y : i16 to i32
    llvm.return %0 : i32
  }

  llvm.func @c02(%x: !llvm.ptr, %y: i8) -> f64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd inbranch uniform(%x : !llvm.ptr)
    %0 = llvm.sitofp %y : i8 to f64
    llvm.return %0 : f64
  }

  // ************************************
  // * Linear with a constant parameter *
  // ************************************

  llvm.func @constlinear(%i: i32) -> f64 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    omp.declare_simd notinbranch linear(%i : i32 = %c1 : i32)
    %zero = llvm.mlir.constant(0.0 : f64) : f64
    llvm.return %zero : f64
  }

  // *************************
  // * sincos-like signature *
  // *************************

  llvm.func @sincos(%in: f64, %sin: !llvm.ptr, %cos: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    omp.declare_simd linear(%sin : !llvm.ptr = %c8 : i64,
                            %cos : !llvm.ptr = %c8 : i64)
    llvm.return
  }

  llvm.func @SinCos(%in: f64, %sin: !llvm.ptr, %cos: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    omp.declare_simd linear(%sin : !llvm.ptr = %c8 : i64,
                            %cos : !llvm.ptr = %c16 : i64)
    llvm.return
  }

  // ************************************
  // * linear(val), linear(ref), linear(uval) *
  // ************************************

  // Listing 2 adapted: linear(val) on a sincos-like signature.
  // val modifier on !llvm.ptr -> LinearVal -> mangled as 'L'.
  // NDS=64 (f64 and pointers are 64-bit), no simdlen -> VLEN from NDS.
  llvm.func @sincos_val(%in: f64, %sin: !llvm.ptr, %cos: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    omp.declare_simd linear(val(%sin : !llvm.ptr = %c8 : i64),
                            val(%cos : !llvm.ptr = %c8 : i64))
    llvm.return
  }

  // Listing 3 adapted: linear(ref) on a pointer parameter.
  // ref modifier -> LinearRef -> mangled as 'R'.
  // MTV is false for LinearRef, NDS comes from PBV of args.
  llvm.func @sincos_ref(%in: f64, %sin: !llvm.ptr, %cos: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    omp.declare_simd linear(ref(%sin : !llvm.ptr = %c8 : i64),
                            ref(%cos : !llvm.ptr = %c8 : i64))
    llvm.return
  }

  // linear(uval) — mangled as 'U'.
  // MTV is false for LinearUVal.
  llvm.func @sincos_uval(%in: f64, %sin: !llvm.ptr, %cos: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    omp.declare_simd linear(uval(%sin : !llvm.ptr = %c8 : i64),
                            uval(%cos : !llvm.ptr = %c8 : i64))
    llvm.return
  }

  // Selection of tests based on the examples provided in chapter 5 of
  // the Vector Function ABI specifications for AArch64, at
  // https://developer.arm.com/products/software-development-tools/hpc/arm-compiler-for-hpc/vector-function-abi.

  // Listing 6, p. 19
  llvm.func @foo4(%x: !llvm.ptr, %y: f32) -> i32 attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c0 = llvm.mlir.constant(0 : i32) : i32
    omp.declare_simd linear(%x : !llvm.ptr = %c4 : i64)
                     aligned(%x : !llvm.ptr -> 16 : i64)
                     simdlen(4)
    llvm.return %c0 : i32
  }

  llvm.func @DoRGB(%x: !llvm.struct<(i8, i8, i8)>) -> !llvm.struct<(i8, i8, i8)> attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    omp.declare_simd notinbranch
    llvm.return %x : !llvm.struct<(i8, i8, i8)>
  }

  // ********************************
  // * arg_types for NDS/WDS fix    *
  // ********************************

  // For LinearRef/LinearUVal, MTV=false.  Without arg_types,
  // opaque !llvm.ptr gives LS=sizeof(ptr)=64.  With arg_types
  // = [f64, i32, i32], the ptr params' LS becomes sizeof(i32)=32,
  // so NDS=min(64,32)=32 -> VLEN={2,4} instead of just {2}.
  llvm.func @sincos_ref_lvt(%in: f64, %sin: !llvm.ptr, %cos: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c4 = llvm.mlir.constant(4 : i64) : i64
    omp.declare_simd linear(ref(%sin : !llvm.ptr = %c4 : i64),
                            ref(%cos : !llvm.ptr = %c4 : i64)) {arg_types = [f64, i32, i32]}
    llvm.return
  }

  // Same but without arg_types: LS=64 (ptr) -> NDS=64 -> VLEN={2} only
  llvm.func @sincos_ref_no_lvt(%in: f64, %sin: !llvm.ptr, %cos: !llvm.ptr) attributes {
    target_features = #llvm.target_features<["+neon"]>
  } {
    %c4 = llvm.mlir.constant(4 : i64) : i64
    omp.declare_simd linear(ref(%sin : !llvm.ptr = %c4 : i64),
                            ref(%cos : !llvm.ptr = %c4 : i64))
    llvm.return
  }
}

// CHECK: attributes {{#[0-9]+}} = {
// CHECK-SAME: "_ZGVnM2v_foo"
// CHECK-SAME: "_ZGVnM4v_foo"
// CHECK-SAME: "_ZGVnM8v_foo"
// CHECK-SAME: "_ZGVnN2v_foo"
// CHECK-SAME: "_ZGVnN4v_foo"
// CHECK-SAME: "_ZGVnN8v_foo"
// CHECK-SAME: "target-features"="+neon"
// CHECK-SAME: }
// CHECK-NOT: _ZGVnN6v_foo

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnN2v_bar" "_ZGVnN4v_bar" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnN2v_baz" "_ZGVnN4v_baz" "target-features"="+neon" }

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2v_foo_int" "_ZGVnM4v_foo_int" "_ZGVnM8v_foo_int" "_ZGVnN2v_foo_int" "_ZGVnN4v_foo_int" "_ZGVnN8v_foo_int" "target-features"="+neon" }
// CHECK-NOT: _ZGVnN6v_foo_int

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM16v_simple_8bit" "_ZGVnM8v_simple_8bit" "_ZGVnN16v_simple_8bit" "_ZGVnN8v_simple_8bit" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM4v_simple_16bit" "_ZGVnM8v_simple_16bit" "_ZGVnN4v_simple_16bit" "_ZGVnN8v_simple_16bit" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2v_simple_32bit" "_ZGVnM4v_simple_32bit" "_ZGVnN2v_simple_32bit" "_ZGVnN4v_simple_32bit" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2v_simple_64bit" "_ZGVnN2v_simple_64bit" "target-features"="+neon" }

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM16v_a01" "_ZGVnM32v_a01" "_ZGVnM8v_a01" "_ZGVnN16v_a01" "_ZGVnN32v_a01" "_ZGVnN8v_a01" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2v_a02" "_ZGVnM4v_a02" "_ZGVnM8v_a02" "_ZGVnN2v_a02" "_ZGVnN4v_a02" "_ZGVnN8v_a02" "target-features"="+neon" }

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2v_b01" "_ZGVnM4v_b01" "_ZGVnN2v_b01" "_ZGVnN4v_b01" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM16v_b02" "_ZGVnM8v_b02" "_ZGVnN16v_b02" "_ZGVnN8v_b02" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2v_b03" "_ZGVnN2v_b03" "target-features"="+neon" }

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM4vv_c01" "_ZGVnM8vv_c01" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM16uv_c02" "_ZGVnM8uv_c02" "target-features"="+neon" }

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnN2l_constlinear" "_ZGVnN4l_constlinear" "target-features"="+neon" }

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2vl8l8_sincos" "_ZGVnN2vl8l8_sincos" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2vl8l16_SinCos" "_ZGVnN2vl8l16_SinCos" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2vL8L8_sincos_val" "_ZGVnN2vL8L8_sincos_val" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2vR8R8_sincos_ref" "_ZGVnN2vR8R8_sincos_ref" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2vU8U8_sincos_uval" "_ZGVnN2vU8U8_sincos_uval" "target-features"="+neon" }
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM4l4a16v_foo4" "_ZGVnN4l4a16v_foo4" "target-features"="+neon" }

// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnN2vv_DoRGB" "target-features"="+neon" }

// arg_types gives LS=32 (from i32) -> NDS=32 -> VLEN={2,4}
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2vR4R4_sincos_ref_lvt" "_ZGVnM4vR4R4_sincos_ref_lvt" "_ZGVnN2vR4R4_sincos_ref_lvt" "_ZGVnN4vR4R4_sincos_ref_lvt" "target-features"="+neon" }

// Without arg_types: LS=64 (ptr) -> NDS=64 -> VLEN={2} only
// CHECK-DAG: attributes {{#[0-9]+}} = { "_ZGVnM2vR4R4_sincos_ref_no_lvt" "_ZGVnN2vR4R4_sincos_ref_no_lvt" "target-features"="+neon" }
