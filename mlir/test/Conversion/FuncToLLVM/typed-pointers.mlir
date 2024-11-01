// RUN: mlir-opt -convert-func-to-llvm='use-opaque-pointers=0' -split-input-file %s | FileCheck %s

//CHECK: llvm.func @second_order_arg(!llvm.ptr<func<void ()>>)
func.func private @second_order_arg(%arg0 : () -> ())

//CHECK: llvm.func @second_order_result() -> !llvm.ptr<func<void ()>>
func.func private @second_order_result() -> (() -> ())

//CHECK: llvm.func @second_order_multi_result() -> !llvm.struct<(ptr<func<i32 ()>>, ptr<func<i64 ()>>, ptr<func<f32 ()>>)>
func.func private @second_order_multi_result() -> (() -> (i32), () -> (i64), () -> (f32))

//CHECK: llvm.func @third_order(!llvm.ptr<func<ptr<func<void ()>> (ptr<func<void ()>>)>>) -> !llvm.ptr<func<ptr<func<void ()>> (ptr<func<void ()>>)>>
func.func private @third_order(%arg0 : (() -> ()) -> (() -> ())) -> ((() -> ()) -> (() -> ()))

//CHECK: llvm.func @fifth_order_left(!llvm.ptr<func<void (ptr<func<void (ptr<func<void (ptr<func<void ()>>)>>)>>)>>)
func.func private @fifth_order_left(%arg0: (((() -> ()) -> ()) -> ()) -> ())

//CHECK: llvm.func @fifth_order_right(!llvm.ptr<func<ptr<func<ptr<func<ptr<func<void ()>> ()>> ()>> ()>>)
func.func private @fifth_order_right(%arg0: () -> (() -> (() -> (() -> ()))))

// Check that memrefs are converted to argument packs if appear as function arguments.
// CHECK: llvm.func @memref_call_conv(!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64)
func.func private @memref_call_conv(%arg0: memref<?xf32>)

// Same in nested functions.
// CHECK: llvm.func @memref_call_conv_nested(!llvm.ptr<func<void (ptr<f32>, ptr<f32>, i64, i64, i64)>>)
func.func private @memref_call_conv_nested(%arg0: (memref<?xf32>) -> ())

//CHECK-LABEL: llvm.func @pass_through(%arg0: !llvm.ptr<func<void ()>>) -> !llvm.ptr<func<void ()>> {
func.func @pass_through(%arg0: () -> ()) -> (() -> ()) {
// CHECK-NEXT:  llvm.br ^bb1(%arg0 : !llvm.ptr<func<void ()>>)
  cf.br ^bb1(%arg0 : () -> ())

//CHECK-NEXT: ^bb1(%0: !llvm.ptr<func<void ()>>):
^bb1(%bbarg: () -> ()):
// CHECK-NEXT:  llvm.return %0 : !llvm.ptr<func<void ()>>
  return %bbarg : () -> ()
}

// CHECK-LABEL: llvm.func @indirect_call(%arg0: !llvm.ptr<func<i32 (f32)>>, %arg1: f32) -> i32 {
func.func @indirect_call(%arg0: (f32) -> i32, %arg1: f32) -> i32 {
// CHECK-NEXT:  %0 = llvm.call %arg0(%arg1) : !llvm.ptr<func<i32 (f32)>>, (f32) -> i32
  %0 = call_indirect %arg0(%arg1) : (f32) -> i32
// CHECK-NEXT:  llvm.return %0 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @get_i64() -> i64
func.func private @get_i64() -> (i64)
// CHECK-LABEL: llvm.func @get_f32() -> f32
func.func private @get_f32() -> (f32)
// CHECK-LABEL: llvm.func @get_memref() -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
func.func private @get_memref() -> (memref<42x?x10x?xf32>)

// CHECK-LABEL: llvm.func @multireturn() -> !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)> {
func.func @multireturn() -> (i64, f32, memref<42x?x10x?xf32>) {
^bb0:
// CHECK-NEXT:  {{.*}} = llvm.call @get_i64() : () -> i64
// CHECK-NEXT:  {{.*}} = llvm.call @get_f32() : () -> f32
// CHECK-NEXT:  {{.*}} = llvm.call @get_memref() : () -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
  %0 = call @get_i64() : () -> (i64)
  %1 = call @get_f32() : () -> (f32)
  %2 = call @get_memref() : () -> (memref<42x?x10x?xf32>)
// CHECK-NEXT:  {{.*}} = llvm.mlir.undef : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
// CHECK-NEXT:  llvm.return {{.*}} : !llvm.struct<(i64, f32, struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>)>
  return %0, %1, %2 : i64, f32, memref<42x?x10x?xf32>
}

//===========================================================================//
// Calling convention on returning unranked memrefs.
// IR below produced by running -finalize-memref-to-llvm without opaque
// pointers on calling-convention.mlir
//===========================================================================//

func.func @return_var_memref(%arg0: memref<4x3xf32>) -> memref<*xf32> attributes {llvm.emit_c_interface} {
  %0 = builtin.unrealized_conversion_cast %arg0 : memref<4x3xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %1 = llvm.mlir.constant(1 : index) : i64
  %2 = llvm.alloca %1 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
  llvm.store %0, %2 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
  %3 = llvm.bitcast %2 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
  %4 = llvm.mlir.constant(2 : index) : i64
  %5 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
  %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(i64, ptr<i8>)>
  %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(i64, ptr<i8>)>
  %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
  return %8 : memref<*xf32>
}

// Check that the result memref is passed as parameter
// CHECK-LABEL: @_mlir_ciface_return_var_memref
// CHECK-SAME: (%{{.*}}: !llvm.ptr<struct<(i64, ptr<i8>)>>, %{{.*}}: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>)

func.func @return_two_var_memref(%arg0: memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface} {
  %0 = builtin.unrealized_conversion_cast %arg0 : memref<4x3xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %1 = llvm.mlir.constant(1 : index) : i64
  %2 = llvm.alloca %1 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
  llvm.store %0, %2 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
  %3 = llvm.bitcast %2 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
  %4 = llvm.mlir.constant(2 : index) : i64
  %5 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
  %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(i64, ptr<i8>)>
  %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(i64, ptr<i8>)>
  %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(i64, ptr<i8>)> to memref<*xf32>
  return %8, %8 : memref<*xf32>, memref<*xf32>
}

// Check that the result memrefs are passed as parameter
// CHECK-LABEL: @_mlir_ciface_return_two_var_memref
// CHECK-SAME: (%{{.*}}: !llvm.ptr<struct<(struct<(i64, ptr<i8>)>, struct<(i64, ptr<i8>)>)>>,
// CHECK-SAME: %{{.*}}: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>)

