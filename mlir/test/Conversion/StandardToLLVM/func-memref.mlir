// RUN: mlir-opt -convert-std-to-llvm -split-input-file %s | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -split-input-file %s | FileCheck %s --check-prefix=BAREPTR

// BAREPTR-LABEL: func @check_noalias
// BAREPTR-SAME: %{{.*}}: !llvm.ptr<f32> {llvm.noalias}, %{{.*}}: !llvm.ptr<f32> {llvm.noalias}
func @check_noalias(%static : memref<2xf32> {llvm.noalias}, %other : memref<2xf32> {llvm.noalias}) {
    return
}

// -----

// CHECK-LABEL: func @check_strided_memref_arguments(
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
func @check_strided_memref_arguments(%static: memref<10x20xf32, affine_map<(i,j)->(20 * i + j + 1)>>,
                                     %dynamic : memref<?x?xf32, affine_map<(i,j)[M]->(M * i + j + 1)>>,
                                     %mixed : memref<10x?xf32, affine_map<(i,j)[M]->(M * i + j + 1)>>) {
  return
}

// -----

// CHECK-LABEL: func @memref_index
// CHECK-SAME: %arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>,
// CHECK-SAME: %arg2: i64, %arg3: i64, %arg4: i64)
// CHECK-SAME: -> !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK32-LABEL: func @memref_index
// CHECK32-SAME: %arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>,
// CHECK32-SAME: %arg2: i32, %arg3: i32, %arg4: i32)
// CHECK32-SAME: -> !llvm.struct<(ptr<i32>, ptr<i32>, i32, array<1 x i32>, array<1 x i32>)>
func @memref_index(%arg0: memref<32xindex>) -> memref<32xindex> {
  return %arg0 : memref<32xindex>
}

// -----

// CHECK-LABEL: func @check_arguments
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
func @check_arguments(%static: memref<10x20xf32>, %dynamic : memref<?x?xf32>, %mixed : memref<10x?xf32>) {
  return
}

// -----

// CHECK-LABEL: func @check_static_return
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
// CHECK-SAME: -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-LABEL: func @check_static_return
// BAREPTR-SAME: (%[[arg:.*]]: !llvm.ptr<f32>) -> !llvm.ptr<f32> {
func @check_static_return(%static : memref<32x18xf32>) -> memref<32x18xf32> {
// CHECK:  llvm.return %{{.*}} : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

// BAREPTR: %[[udf:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base0:.*]] = llvm.insertvalue %[[arg]], %[[udf]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[aligned:.*]] = llvm.insertvalue %[[arg]], %[[base0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val0:.*]] = llvm.mlir.constant(0 : index) : i64
// BAREPTR-NEXT: %[[ins0:.*]] = llvm.insertvalue %[[val0]], %[[aligned]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val1:.*]] = llvm.mlir.constant(32 : index) : i64
// BAREPTR-NEXT: %[[ins1:.*]] = llvm.insertvalue %[[val1]], %[[ins0]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val2:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins2:.*]] = llvm.insertvalue %[[val2]], %[[ins1]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val3:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins3:.*]] = llvm.insertvalue %[[val3]], %[[ins2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val4:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT: %[[ins4:.*]] = llvm.insertvalue %[[val4]], %[[ins3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base1:.*]] = llvm.extractvalue %[[ins4]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: llvm.return %[[base1]] : !llvm.ptr<f32>
  return %static : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @check_static_return_with_offset
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
// CHECK-SAME: -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-LABEL: func @check_static_return_with_offset
// BAREPTR-SAME: (%[[arg:.*]]: !llvm.ptr<f32>) -> !llvm.ptr<f32> {
func @check_static_return_with_offset(%static : memref<32x18xf32, offset:7, strides:[22,1]>) -> memref<32x18xf32, offset:7, strides:[22,1]> {
// CHECK:  llvm.return %{{.*}} : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

// BAREPTR: %[[udf:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base0:.*]] = llvm.insertvalue %[[arg]], %[[udf]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[aligned:.*]] = llvm.insertvalue %[[arg]], %[[base0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val0:.*]] = llvm.mlir.constant(7 : index) : i64
// BAREPTR-NEXT: %[[ins0:.*]] = llvm.insertvalue %[[val0]], %[[aligned]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val1:.*]] = llvm.mlir.constant(32 : index) : i64
// BAREPTR-NEXT: %[[ins1:.*]] = llvm.insertvalue %[[val1]], %[[ins0]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val2:.*]] = llvm.mlir.constant(22 : index) : i64
// BAREPTR-NEXT: %[[ins2:.*]] = llvm.insertvalue %[[val2]], %[[ins1]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val3:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins3:.*]] = llvm.insertvalue %[[val3]], %[[ins2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val4:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT: %[[ins4:.*]] = llvm.insertvalue %[[val4]], %[[ins3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base1:.*]] = llvm.extractvalue %[[ins4]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: llvm.return %[[base1]] : !llvm.ptr<f32>
  return %static : memref<32x18xf32, offset:7, strides:[22,1]>
}

// -----

// BAREPTR: llvm.func @foo(!llvm.ptr<i8>) -> !llvm.ptr<i8>
func private @foo(memref<10xi8>) -> memref<20xi8>

// BAREPTR-LABEL: func @check_memref_func_call
// BAREPTR-SAME:    %[[in:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
func @check_memref_func_call(%in : memref<10xi8>) -> memref<20xi8> {
  // BAREPTR:         %[[inDesc:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0]
  // BAREPTR-NEXT:    %[[barePtr:.*]] = llvm.extractvalue %[[inDesc]][1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[call:.*]] = llvm.call @foo(%[[barePtr]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  // BAREPTR-NEXT:    %[[desc0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[desc1:.*]] = llvm.insertvalue %[[call]], %[[desc0]][0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[desc2:.*]] = llvm.insertvalue %[[call]], %[[desc1]][1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
  // BAREPTR-NEXT:    %[[desc4:.*]] = llvm.insertvalue %[[c0]], %[[desc2]][2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c20:.*]] = llvm.mlir.constant(20 : index) : i64
  // BAREPTR-NEXT:    %[[desc6:.*]] = llvm.insertvalue %[[c20]], %[[desc4]][3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
  // BAREPTR-NEXT:    %[[outDesc:.*]] = llvm.insertvalue %[[c1]], %[[desc6]][4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  %res = call @foo(%in) : (memref<10xi8>) -> (memref<20xi8>)
  // BAREPTR-NEXT:    %[[res:.*]] = llvm.extractvalue %[[outDesc]][1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    llvm.return %[[res]] : !llvm.ptr<i8>
  return %res : memref<20xi8>
}

// -----

// Unranked memrefs are currently not supported in the bare-ptr calling
// convention. Check that the conversion to the LLVM-IR dialect doesn't happen
// in the presence of unranked memrefs when using such a calling convention.

// BAREPTR: func private @hoo(memref<*xi8>) -> memref<*xi8>
func private @hoo(memref<*xi8>) -> memref<*xi8>

// BAREPTR-LABEL: func @check_unranked_memref_func_call(%{{.*}}: memref<*xi8>) -> memref<*xi8>
func @check_unranked_memref_func_call(%in: memref<*xi8>) -> memref<*xi8> {
  // BAREPTR-NEXT: call @hoo(%{{.*}}) : (memref<*xi8>) -> memref<*xi8>
  %res = call @hoo(%in) : (memref<*xi8>) -> memref<*xi8>
  // BAREPTR-NEXT: return %{{.*}} : memref<*xi8>
  return %res : memref<*xi8>
}

// -----

// Should not convert memrefs with unsupported types in any convention.

// CHECK: @unsupported_memref_element_type
// CHECK-SAME: memref<
// CHECK-NOT: !llvm.struct
// BAREPTR: @unsupported_memref_element_type
// BAREPTR-SAME: memref<
// BAREPTR-NOT: !llvm.ptr
func private @unsupported_memref_element_type() -> memref<42 x !test.memref_element>

// CHECK: @unsupported_unranked_memref_element_type
// CHECK-SAME: memref<
// CHECK-NOT: !llvm.struct
// BAREPTR: @unsupported_unranked_memref_element_type
// BAREPTR-SAME: memref<
// BAREPTR-NOT: !llvm.ptr
func private @unsupported_unranked_memref_element_type() -> memref<* x !test.memref_element>

// -----

// BAREPTR: llvm.func @goo(f32) -> f32
func private @goo(f32) -> f32

// BAREPTR-LABEL: func @check_scalar_func_call
// BAREPTR-SAME:    %[[in:.*]]: f32)
func @check_scalar_func_call(%in : f32) {
  // BAREPTR-NEXT:    %[[call:.*]] = llvm.call @goo(%[[in]]) : (f32) -> f32
  %res = call @goo(%in) : (f32) -> (f32)
  return
}

// -----

!base_type = type memref<64xi32, 201>

// CHECK-LABEL: func @loop_carried
// BAREPTR-LABEL: func @loop_carried
func @loop_carried(%arg0 : index, %arg1 : index, %arg2 : index, %base0 : !base_type, %base1 : !base_type) -> (!base_type, !base_type) {
  // This test checks that in the BAREPTR case, the branch arguments only forward the descriptor.
  // This test was lowered from a simple scf.for that swaps 2 memref iter_args.
  //      BAREPTR: llvm.br ^bb1(%{{.*}}, %{{.*}}, %{{.*}} : i64, !llvm.struct<(ptr<i32, 201>, ptr<i32, 201>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr<i32, 201>, ptr<i32, 201>, i64, array<1 x i64>, array<1 x i64>)>)
  br ^bb1(%arg0, %base0, %base1 : index, memref<64xi32, 201>, memref<64xi32, 201>)

  // BAREPTR-NEXT: ^bb1
  // BAREPTR-NEXT:   llvm.icmp
  // BAREPTR-NEXT:   llvm.cond_br %{{.*}}, ^bb2, ^bb3
  ^bb1(%0: index, %1: memref<64xi32, 201>, %2: memref<64xi32, 201>):  // 2 preds: ^bb0, ^bb2
    %3 = cmpi slt, %0, %arg1 : index
    cond_br %3, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %4 = addi %0, %arg2 : index
    br ^bb1(%4, %2, %1 : index, memref<64xi32, 201>, memref<64xi32, 201>)
  ^bb3:  // pred: ^bb1
    return %1, %2 : memref<64xi32, 201>, memref<64xi32, 201>
}
