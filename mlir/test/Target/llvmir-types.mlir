// RUN: mlir-translate -test-mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @primitives() {
  // CHECK: declare void @return_void()
  // CHECK: declare void @return_void_round()
  "llvm.test_introduce_func"() { name = "return_void", type = !llvm2.void } : () -> ()
  // CHECK: declare half @return_half()
  // CHECK: declare half @return_half_round()
  "llvm.test_introduce_func"() { name = "return_half", type = !llvm2.half } : () -> ()
  // CHECK: declare bfloat @return_bfloat()
  // CHECK: declare bfloat @return_bfloat_round()
  "llvm.test_introduce_func"() { name = "return_bfloat", type = !llvm2.bfloat } : () -> ()
  // CHECK: declare float @return_float()
  // CHECK: declare float @return_float_round()
  "llvm.test_introduce_func"() { name = "return_float", type = !llvm2.float } : () -> ()
  // CHECK: declare double @return_double()
  // CHECK: declare double @return_double_round()
  "llvm.test_introduce_func"() { name = "return_double", type = !llvm2.double } : () -> ()
  // CHECK: declare fp128 @return_fp128()
  // CHECK: declare fp128 @return_fp128_round()
  "llvm.test_introduce_func"() { name = "return_fp128", type = !llvm2.fp128 } : () -> ()
  // CHECK: declare x86_fp80 @return_x86_fp80()
  // CHECK: declare x86_fp80 @return_x86_fp80_round()
  "llvm.test_introduce_func"() { name = "return_x86_fp80", type = !llvm2.x86_fp80 } : () -> ()
  // CHECK: declare ppc_fp128 @return_ppc_fp128()
  // CHECK: declare ppc_fp128 @return_ppc_fp128_round()
  "llvm.test_introduce_func"() { name = "return_ppc_fp128", type = !llvm2.ppc_fp128 } : () -> ()
  // CHECK: declare x86_mmx @return_x86_mmx()
  // CHECK: declare x86_mmx @return_x86_mmx_round()
  "llvm.test_introduce_func"() { name = "return_x86_mmx", type = !llvm2.x86_mmx } : () -> ()
  llvm.return
}

llvm.func @funcs() {
  // CHECK: declare void @f_void_i32(i32)
  // CHECK: declare void @f_void_i32_round(i32)
  "llvm.test_introduce_func"() { name ="f_void_i32", type = !llvm2.func<void (i32)> } : () -> ()
  // CHECK: declare i32 @f_i32_empty()
  // CHECK: declare i32 @f_i32_empty_round()
  "llvm.test_introduce_func"() { name ="f_i32_empty", type = !llvm2.func<i32 ()> } : () -> ()
  // CHECK: declare i32 @f_i32_half_bfloat_float_double(half, bfloat, float, double)
  // CHECK: declare i32 @f_i32_half_bfloat_float_double_round(half, bfloat, float, double)
  "llvm.test_introduce_func"() { name ="f_i32_half_bfloat_float_double", type = !llvm2.func<i32 (half, bfloat, float, double)> } : () -> ()
  // CHECK: declare i32 @f_i32_i32_i32(i32, i32)
  // CHECK: declare i32 @f_i32_i32_i32_round(i32, i32)
  "llvm.test_introduce_func"() { name ="f_i32_i32_i32", type = !llvm2.func<i32 (i32, i32)> } : () -> ()
  // CHECK: declare void @f_void_variadic(...)
  // CHECK: declare void @f_void_variadic_round(...)
  "llvm.test_introduce_func"() { name ="f_void_variadic", type = !llvm2.func<void (...)> } : () -> ()
  // CHECK: declare void @f_void_i32_i32_variadic(i32, i32, ...)
  // CHECK: declare void @f_void_i32_i32_variadic_round(i32, i32, ...)
  "llvm.test_introduce_func"() { name ="f_void_i32_i32_variadic", type = !llvm2.func<void (i32, i32, ...)> } : () -> ()
  llvm.return
}

llvm.func @ints() {
  // CHECK: declare i1 @return_i1()
  // CHECK: declare i1 @return_i1_round()
  "llvm.test_introduce_func"() { name = "return_i1", type = !llvm2.i1 } : () -> ()
  // CHECK: declare i8 @return_i8()
  // CHECK: declare i8 @return_i8_round()
  "llvm.test_introduce_func"() { name = "return_i8", type = !llvm2.i8 } : () -> ()
  // CHECK: declare i16 @return_i16()
  // CHECK: declare i16 @return_i16_round()
  "llvm.test_introduce_func"() { name = "return_i16", type = !llvm2.i16 } : () -> ()
  // CHECK: declare i32 @return_i32()
  // CHECK: declare i32 @return_i32_round()
  "llvm.test_introduce_func"() { name = "return_i32", type = !llvm2.i32 } : () -> ()
  // CHECK: declare i64 @return_i64()
  // CHECK: declare i64 @return_i64_round()
  "llvm.test_introduce_func"() { name = "return_i64", type = !llvm2.i64 } : () -> ()
  // CHECK: declare i57 @return_i57()
  // CHECK: declare i57 @return_i57_round()
  "llvm.test_introduce_func"() { name = "return_i57", type = !llvm2.i57 } : () -> ()
  // CHECK: declare i129 @return_i129()
  // CHECK: declare i129 @return_i129_round()
  "llvm.test_introduce_func"() { name = "return_i129", type = !llvm2.i129 } : () -> ()
  llvm.return
}

llvm.func @pointers() {
  // CHECK: declare i8* @return_pi8()
  // CHECK: declare i8* @return_pi8_round()
  "llvm.test_introduce_func"() { name = "return_pi8", type = !llvm2.ptr<i8> } : () -> ()
  // CHECK: declare float* @return_pfloat()
  // CHECK: declare float* @return_pfloat_round()
  "llvm.test_introduce_func"() { name = "return_pfloat", type = !llvm2.ptr<float> } : () -> ()
  // CHECK: declare i8** @return_ppi8()
  // CHECK: declare i8** @return_ppi8_round()
  "llvm.test_introduce_func"() { name = "return_ppi8", type = !llvm2.ptr<ptr<i8>> } : () -> ()
  // CHECK: declare i8***** @return_pppppi8()
  // CHECK: declare i8***** @return_pppppi8_round()
  "llvm.test_introduce_func"() { name = "return_pppppi8", type = !llvm2.ptr<ptr<ptr<ptr<ptr<i8>>>>> } : () -> ()
  // CHECK: declare i8* @return_pi8_0()
  // CHECK: declare i8* @return_pi8_0_round()
  "llvm.test_introduce_func"() { name = "return_pi8_0", type = !llvm2.ptr<i8, 0> } : () -> ()
  // CHECK: declare i8 addrspace(1)* @return_pi8_1()
  // CHECK: declare i8 addrspace(1)* @return_pi8_1_round()
  "llvm.test_introduce_func"() { name = "return_pi8_1", type = !llvm2.ptr<i8, 1> } : () -> ()
  // CHECK: declare i8 addrspace(42)* @return_pi8_42()
  // CHECK: declare i8 addrspace(42)* @return_pi8_42_round()
  "llvm.test_introduce_func"() { name = "return_pi8_42", type = !llvm2.ptr<i8, 42> } : () -> ()
  // CHECK: declare i8 addrspace(42)* addrspace(9)* @return_ppi8_42_9()
  // CHECK: declare i8 addrspace(42)* addrspace(9)* @return_ppi8_42_9_round()
  "llvm.test_introduce_func"() { name = "return_ppi8_42_9", type = !llvm2.ptr<ptr<i8, 42>, 9> } : () -> ()
  llvm.return
}

llvm.func @vectors() {
  // CHECK: declare <4 x i32> @return_v4_i32()
  // CHECK: declare <4 x i32> @return_v4_i32_round()
  "llvm.test_introduce_func"() { name = "return_v4_i32", type = !llvm2.vec<4 x i32> } : () -> ()
  // CHECK: declare <4 x float> @return_v4_float()
  // CHECK: declare <4 x float> @return_v4_float_round()
  "llvm.test_introduce_func"() { name = "return_v4_float", type = !llvm2.vec<4 x float> } : () -> ()
  // CHECK: declare <vscale x 4 x i32> @return_vs_4_i32()
  // CHECK: declare <vscale x 4 x i32> @return_vs_4_i32_round()
  "llvm.test_introduce_func"() { name = "return_vs_4_i32", type = !llvm2.vec<? x 4 x i32> } : () -> ()
  // CHECK: declare <vscale x 8 x half> @return_vs_8_half()
  // CHECK: declare <vscale x 8 x half> @return_vs_8_half_round()
  "llvm.test_introduce_func"() { name = "return_vs_8_half", type = !llvm2.vec<? x 8 x half> } : () -> ()
  // CHECK: declare <4 x i8*> @return_v_4_pi8()
  // CHECK: declare <4 x i8*> @return_v_4_pi8_round()
  "llvm.test_introduce_func"() { name = "return_v_4_pi8", type = !llvm2.vec<4 x ptr<i8>> } : () -> ()
  llvm.return
}

llvm.func @arrays() {
  // CHECK: declare [10 x i32] @return_a10_i32()
  // CHECK: declare [10 x i32] @return_a10_i32_round()
  "llvm.test_introduce_func"() { name = "return_a10_i32", type = !llvm2.array<10 x i32> } : () -> ()
  // CHECK: declare [8 x float] @return_a8_float()
  // CHECK: declare [8 x float] @return_a8_float_round()
  "llvm.test_introduce_func"() { name = "return_a8_float", type = !llvm2.array<8 x float> } : () -> ()
  // CHECK: declare [10 x i32 addrspace(4)*] @return_a10_pi32_4()
  // CHECK: declare [10 x i32 addrspace(4)*] @return_a10_pi32_4_round()
  "llvm.test_introduce_func"() { name = "return_a10_pi32_4", type = !llvm2.array<10 x ptr<i32, 4>> } : () -> ()
  // CHECK: declare [10 x [4 x float]] @return_a10_a4_float()
  // CHECK: declare [10 x [4 x float]] @return_a10_a4_float_round()
  "llvm.test_introduce_func"() { name = "return_a10_a4_float", type = !llvm2.array<10 x array<4 x float>> } : () -> ()
  llvm.return
}

llvm.func @literal_structs() {
  // CHECK: declare {} @return_struct_empty()
  // CHECK: declare {} @return_struct_empty_round()
  "llvm.test_introduce_func"() { name = "return_struct_empty", type = !llvm2.struct<()> } : () -> ()
  // CHECK: declare { i32 } @return_s_i32()
  // CHECK: declare { i32 } @return_s_i32_round()
  "llvm.test_introduce_func"() { name = "return_s_i32", type = !llvm2.struct<(i32)> } : () -> ()
  // CHECK: declare { float, i32 } @return_s_float_i32()
  // CHECK: declare { float, i32 } @return_s_float_i32_round()
  "llvm.test_introduce_func"() { name = "return_s_float_i32", type = !llvm2.struct<(float, i32)> } : () -> ()
  // CHECK: declare { { i32 } } @return_s_s_i32()
  // CHECK: declare { { i32 } } @return_s_s_i32_round()
  "llvm.test_introduce_func"() { name = "return_s_s_i32", type = !llvm2.struct<(struct<(i32)>)> } : () -> ()
  // CHECK: declare { i32, { i32 }, float } @return_s_i32_s_i32_float()
  // CHECK: declare { i32, { i32 }, float } @return_s_i32_s_i32_float_round()
  "llvm.test_introduce_func"() { name = "return_s_i32_s_i32_float", type = !llvm2.struct<(i32, struct<(i32)>, float)> } : () -> ()

  // CHECK: declare <{}> @return_sp_empty()
  // CHECK: declare <{}> @return_sp_empty_round()
  "llvm.test_introduce_func"() { name = "return_sp_empty", type = !llvm2.struct<packed ()> } : () -> ()
  // CHECK: declare <{ i32 }> @return_sp_i32()
  // CHECK: declare <{ i32 }> @return_sp_i32_round()
  "llvm.test_introduce_func"() { name = "return_sp_i32", type = !llvm2.struct<packed (i32)> } : () -> ()
  // CHECK: declare <{ float, i32 }> @return_sp_float_i32()
  // CHECK: declare <{ float, i32 }> @return_sp_float_i32_round()
  "llvm.test_introduce_func"() { name = "return_sp_float_i32", type = !llvm2.struct<packed (float, i32)> } : () -> ()
  // CHECK: declare <{ i32, { i32, i1 }, float }> @return_sp_i32_s_i31_1_float()
  // CHECK: declare <{ i32, { i32, i1 }, float }> @return_sp_i32_s_i31_1_float_round()
  "llvm.test_introduce_func"() { name = "return_sp_i32_s_i31_1_float", type = !llvm2.struct<packed (i32, struct<(i32, i1)>, float)> } : () -> ()

  // CHECK: declare { <{ i32 }> } @return_s_sp_i32()
  // CHECK: declare { <{ i32 }> } @return_s_sp_i32_round()
  "llvm.test_introduce_func"() { name = "return_s_sp_i32", type = !llvm2.struct<(struct<packed (i32)>)> } : () -> ()
  // CHECK: declare <{ { i32 } }> @return_sp_s_i32()
  // CHECK: declare <{ { i32 } }> @return_sp_s_i32_round()
  "llvm.test_introduce_func"() { name = "return_sp_s_i32", type = !llvm2.struct<packed (struct<(i32)>)> } : () -> ()
  llvm.return
}

// -----
// Put structs into a separate split so that we can match their declarations
// locally.

// CHECK: %empty = type {}
// CHECK: %opaque = type opaque
// CHECK: %long = type { i32, { i32, i1 }, float, void ()* }
// CHECK: %self-recursive = type { %self-recursive* }
// CHECK: %unpacked = type { i32 }
// CHECK: %packed = type <{ i32 }>
// CHECK: %"name with spaces and !^$@$#" = type <{ i32 }>
// CHECK: %mutually-a = type { %mutually-b* }
// CHECK: %mutually-b = type { %mutually-a addrspace(3)* }
// CHECK: %struct-of-arrays = type { [10 x i32] }
// CHECK: %array-of-structs = type { i32 }
// CHECK: %ptr-to-struct = type { i8 }

llvm.func @identified_structs() {
  // CHECK: declare %empty
  "llvm.test_introduce_func"() { name = "return_s_empty", type = !llvm2.struct<"empty", ()> } : () -> ()
  // CHECK: declare %opaque
  "llvm.test_introduce_func"() { name = "return_s_opaque", type = !llvm2.struct<"opaque", opaque> } : () -> ()
  // CHECK: declare %long
  "llvm.test_introduce_func"() { name = "return_s_long", type = !llvm2.struct<"long", (i32, struct<(i32, i1)>, float, ptr<func<void ()>>)> } : () -> ()
  // CHECK: declare %self-recursive
  "llvm.test_introduce_func"() { name = "return_s_self_recurisve", type = !llvm2.struct<"self-recursive", (ptr<struct<"self-recursive">>)> } : () -> ()
  // CHECK: declare %unpacked
  "llvm.test_introduce_func"() { name = "return_s_unpacked", type = !llvm2.struct<"unpacked", (i32)> } : () -> ()
  // CHECK: declare %packed
  "llvm.test_introduce_func"() { name = "return_s_packed", type = !llvm2.struct<"packed", packed (i32)> } : () -> ()
  // CHECK: declare %"name with spaces and !^$@$#"
  "llvm.test_introduce_func"() { name = "return_s_symbols", type = !llvm2.struct<"name with spaces and !^$@$#", packed (i32)> } : () -> ()

  // CHECK: declare %mutually-a
  "llvm.test_introduce_func"() { name = "return_s_mutually_a", type = !llvm2.struct<"mutually-a", (ptr<struct<"mutually-b", (ptr<struct<"mutually-a">, 3>)>>)> } : () -> ()
  // CHECK: declare %mutually-b
  "llvm.test_introduce_func"() { name = "return_s_mutually_b", type = !llvm2.struct<"mutually-b", (ptr<struct<"mutually-a", (ptr<struct<"mutually-b">>)>, 3>)> } : () -> ()

  // CHECK: declare %struct-of-arrays
  "llvm.test_introduce_func"() { name = "return_s_struct_of_arrays", type = !llvm2.struct<"struct-of-arrays", (array<10 x i32>)> } : () -> ()
  // CHECK: declare [10 x %array-of-structs]
  "llvm.test_introduce_func"() { name = "return_s_array_of_structs", type = !llvm2.array<10 x struct<"array-of-structs", (i32)>> } : () -> ()
  // CHECK: declare %ptr-to-struct*
  "llvm.test_introduce_func"() { name = "return_s_ptr_to_struct", type = !llvm2.ptr<struct<"ptr-to-struct", (i8)>> } : () -> ()
  llvm.return
}
