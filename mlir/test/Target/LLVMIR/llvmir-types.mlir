// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

//
// Primitives.
//

// CHECK: declare void @return_void()
llvm.func @return_void() -> !llvm.void
// CHECK: declare half @return_half()
llvm.func @return_half() -> f16
// CHECK: declare bfloat @return_bfloat()
llvm.func @return_bfloat() -> bf16
// CHECK: declare float @return_float()
llvm.func @return_float() -> f32
// CHECK: declare double @return_double()
llvm.func @return_double() -> f64
// CHECK: declare fp128 @return_fp128()
llvm.func @return_fp128() -> f128
// CHECK: declare x86_fp80 @return_x86_fp80()
llvm.func @return_x86_fp80() -> f80
// CHECK: declare ppc_fp128 @return_ppc_fp128()
llvm.func @return_ppc_fp128() -> !llvm.ppc_fp128
// CHECK: declare x86_mmx @return_x86_mmx()
llvm.func @return_x86_mmx() -> !llvm.x86_mmx

//
// Functions.
//

// CHECK: declare void @f_void_i32(i32)
llvm.func @f_void_i32(i32) -> !llvm.void
// CHECK: declare i32 @f_i32_empty()
llvm.func @f_i32_empty() -> i32
// CHECK: declare i32 @f_i32_half_bfloat_float_double(half, bfloat, float, double)
llvm.func @f_i32_half_bfloat_float_double(f16, bf16, f32, f64) -> i32
// CHECK: declare i32 @f_i32_i32_i32(i32, i32)
llvm.func @f_i32_i32_i32(i32, i32) -> i32
// CHECK: declare void @f_void_variadic(...)
llvm.func @f_void_variadic(...)
// CHECK: declare void @f_void_i32_i32_variadic(i32, i32, ...)
llvm.func @f_void_i32_i32_variadic(i32, i32, ...)
// CHECK: declare ptr @f_f_i32_i32()
llvm.func @f_f_i32_i32() -> !llvm.ptr

//
// Integers.
//

// CHECK: declare i1 @return_i1()
llvm.func @return_i1() -> i1
// CHECK: declare i8 @return_i8()
llvm.func @return_i8() -> i8
// CHECK: declare i16 @return_i16()
llvm.func @return_i16() -> i16
// CHECK: declare i32 @return_i32()
llvm.func @return_i32() -> i32
// CHECK: declare i64 @return_i64()
llvm.func @return_i64() -> i64
// CHECK: declare i57 @return_i57()
llvm.func @return_i57() -> i57
// CHECK: declare i129 @return_i129()
llvm.func @return_i129() -> i129

//
// Pointers.
//

// CHECK: declare ptr @return_p()
llvm.func @return_p() -> !llvm.ptr
// CHECK: declare ptr addrspace(1) @return_p_1()
llvm.func @return_p_1() -> !llvm.ptr<1>
// CHECK: declare ptr addrspace(42) @return_p_42()
llvm.func @return_p_42() -> !llvm.ptr<42>

//
// Vectors.
//

// CHECK: declare <4 x i32> @return_v4_i32()
llvm.func @return_v4_i32() -> vector<4xi32>
// CHECK: declare <4 x float> @return_v4_float()
llvm.func @return_v4_float() -> vector<4xf32>
// CHECK: declare <vscale x 4 x float> @return_vs_4_float()
llvm.func @return_vs_4_float() -> vector<[4]xf32>
// CHECK: declare <vscale x 4 x i32> @return_vs_4_i32()
llvm.func @return_vs_4_i32() -> !llvm.vec<?x4 x i32>
// CHECK: declare <vscale x 8 x half> @return_vs_8_half()
llvm.func @return_vs_8_half() -> !llvm.vec<?x8 x f16>
// CHECK: declare <4 x ptr> @return_v_4_pi8()
llvm.func @return_v_4_pi8() -> !llvm.vec<4xptr>

//
// Arrays.
//

// CHECK: declare [10 x i32] @return_a10_i32()
llvm.func @return_a10_i32() -> !llvm.array<10 x i32>
// CHECK: declare [8 x float] @return_a8_float()
llvm.func @return_a8_float() -> !llvm.array<8 x f32>
// CHECK: declare [10 x ptr addrspace(4)] @return_a10_p_4()
llvm.func @return_a10_p_4() -> !llvm.array<10 x ptr<4>>
// CHECK: declare [10 x [4 x float]] @return_a10_a4_float()
llvm.func @return_a10_a4_float() -> !llvm.array<10 x array<4 x f32>>

//
// Literal structures.
//

// CHECK: declare {} @return_struct_empty()
llvm.func @return_struct_empty() -> !llvm.struct<()>
// CHECK: declare { i32 } @return_s_i32()
llvm.func @return_s_i32() -> !llvm.struct<(i32)>
// CHECK: declare { float, i32 } @return_s_float_i32()
llvm.func @return_s_float_i32() -> !llvm.struct<(f32, i32)>
// CHECK: declare { { i32 } } @return_s_s_i32()
llvm.func @return_s_s_i32() -> !llvm.struct<(struct<(i32)>)>
// CHECK: declare { i32, { i32 }, float } @return_s_i32_s_i32_float()
llvm.func @return_s_i32_s_i32_float() -> !llvm.struct<(i32, struct<(i32)>, f32)>

// CHECK: declare <{}> @return_sp_empty()
llvm.func @return_sp_empty() -> !llvm.struct<packed ()>
// CHECK: declare <{ i32 }> @return_sp_i32()
llvm.func @return_sp_i32() -> !llvm.struct<packed (i32)>
// CHECK: declare <{ float, i32 }> @return_sp_float_i32()
llvm.func @return_sp_float_i32() -> !llvm.struct<packed (f32, i32)>
// CHECK: declare <{ i32, { i32, i1 }, float }> @return_sp_i32_s_i31_1_float()
llvm.func @return_sp_i32_s_i31_1_float() -> !llvm.struct<packed (i32, struct<(i32, i1)>, f32)>

// CHECK: declare { <{ i32 }> } @return_s_sp_i32()
llvm.func @return_s_sp_i32() -> !llvm.struct<(struct<packed (i32)>)>
// CHECK: declare <{ { i32 } }> @return_sp_s_i32()
llvm.func @return_sp_s_i32() -> !llvm.struct<packed (struct<(i32)>)>

// CHECK: declare target("target-no-param") @return_target_ext_no_param()
llvm.func @return_target_ext_no_param() -> !llvm.target<"target-no-param">

// CHECK: declare target("target-type-param", i32, double) @return_target_ext_type_params()
llvm.func @return_target_ext_type_params() -> !llvm.target<"target-type-param", i32, f64>

// CHECK: declare target("target-int-param", 0, 42) @return_target_ext_int_params()
llvm.func @return_target_ext_int_params() -> !llvm.target<"target-int-param", 0, 42>

// CHECK: declare target("target-params", i32, double, 0, 5) @return_target_ext_params()
llvm.func @return_target_ext_params() -> !llvm.target<"target-params", i32, f64, 0, 5>

// -----
// Put structs into a separate split so that we can match their declarations
// locally.

// CHECK: %empty = type {}
// CHECK: %opaque = type opaque
// CHECK: %long = type { i32, { i32, i1 }, float, ptr }
// CHECK: %unpacked = type { i32 }
// CHECK: %packed = type <{ i32 }>
// CHECK: %"name with spaces and !^$@$#" = type <{ i32 }>
// CHECK: %struct-of-arrays = type { [10 x i32] }
// CHECK: %array-of-structs = type { i32 }

// CHECK: declare %empty
llvm.func @return_s_empty() -> !llvm.struct<"empty", ()>
// CHECK: declare %opaque
llvm.func @return_s_opaque() -> !llvm.struct<"opaque", opaque>
// CHECK: declare %long
llvm.func @return_s_long() -> !llvm.struct<"long", (i32, struct<(i32, i1)>, f32, ptr)>
// CHECK: declare %unpacked
llvm.func @return_s_unpacked() -> !llvm.struct<"unpacked", (i32)>
// CHECK: declare %packed
llvm.func @return_s_packed() -> !llvm.struct<"packed", packed (i32)>
// CHECK: declare %"name with spaces and !^$@$#"
llvm.func @return_s_symbols() -> !llvm.struct<"name with spaces and !^$@$#", packed (i32)>

// CHECK: declare %struct-of-arrays
llvm.func @return_s_struct_of_arrays() -> !llvm.struct<"struct-of-arrays", (array<10 x i32>)>
// CHECK: declare [10 x %array-of-structs]
llvm.func @return_s_array_of_structs() -> !llvm.array<10 x struct<"array-of-structs", (i32)>>
