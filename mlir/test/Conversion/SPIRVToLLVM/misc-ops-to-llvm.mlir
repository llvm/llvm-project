// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.CompositeExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @composite_extract_array
spv.func @composite_extract_array(%arg: !spv.array<4x!spv.array<4xf32>>) "None" {
  // CHECK: llvm.extractvalue %{{.*}}[1, 3] : !llvm.array<4 x array<4 x f32>>
  %0 = spv.CompositeExtract %arg[1 : i32, 3 : i32] : !spv.array<4x!spv.array<4xf32>>
  spv.Return
}

// CHECK-LABEL: @composite_extract_vector
spv.func @composite_extract_vector(%arg: vector<3xf32>) "None" {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.extractelement %{{.*}}[%[[ZERO]] : i32] : vector<3xf32>
  %0 = spv.CompositeExtract %arg[0 : i32] : vector<3xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.CompositeInsert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @composite_insert_struct
spv.func @composite_insert_struct(%arg0: i32, %arg1: !spv.struct<(f32, !spv.array<4xi32>)>) "None" {
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1, 3] : !llvm.struct<packed (f32, array<4 x i32>)>
  %0 = spv.CompositeInsert %arg0, %arg1[1 : i32, 3 : i32] : i32 into !spv.struct<(f32, !spv.array<4xi32>)>
  spv.Return
}

// CHECK-LABEL: @composite_insert_vector
spv.func @composite_insert_vector(%arg0: vector<3xf32>, %arg1: f32) "None" {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.insertelement %{{.*}}, %{{.*}}[%[[ONE]] : i32] : vector<3xf32>
  %0 = spv.CompositeInsert %arg1, %arg0[1 : i32] : f32 into vector<3xf32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.Select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @select_scalar
spv.func @select_scalar(%arg0: i1, %arg1: vector<3xi32>, %arg2: f32) "None" {
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, vector<3xi32>
  %0 = spv.Select %arg0, %arg1, %arg1 : i1, vector<3xi32>
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, f32
  %1 = spv.Select %arg0, %arg2, %arg2 : i1, f32
  spv.Return
}

// CHECK-LABEL: @select_vector
spv.func @select_vector(%arg0: vector<2xi1>, %arg1: vector<2xi32>) "None" {
  // CHECK: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : vector<2xi1>, vector<2xi32>
  %0 = spv.Select %arg0, %arg1, %arg1 : vector<2xi1>, vector<2xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.VectorShuffle
//===----------------------------------------------------------------------===//

spv.func @vector_shuffle_same_size(%vector1: vector<2xf32>, %vector2: vector<2xf32>) -> vector<3xf32> "None" {
  //      CHECK: %[[res:.*]] = llvm.shufflevector {{.*}} [0 : i32, 2 : i32, -1 : i32] : vector<2xf32>, vector<2xf32>
  // CHECK-NEXT: return %[[res]] : vector<3xf32>
  %0 = spv.VectorShuffle [0: i32, 2: i32, 0xffffffff: i32] %vector1: vector<2xf32>, %vector2: vector<2xf32> -> vector<3xf32>
  spv.ReturnValue %0: vector<3xf32>
}

spv.func @vector_shuffle_different_size(%vector1: vector<3xf32>, %vector2: vector<2xf32>) -> vector<3xf32> "None" {
  //      CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<3xf32>
  // CHECK-NEXT: %[[C0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT: %[[EXT0:.*]] = llvm.extractelement %arg0[%[[C0_1]] : i32] : vector<3xf32>
  // CHECK-NEXT: %[[INSERT0:.*]] = llvm.insertelement %[[EXT0]], %[[UNDEF]][%[[C0_0]] : i32] : vector<3xf32>
  // CHECK-NEXT: %[[C1_0:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[C1_1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[EXT1:.*]] = llvm.extractelement {{.*}}[%[[C1_1]] : i32] : vector<2xf32>
  // CHECK-NEXT: %[[RES:.*]] = llvm.insertelement %[[EXT1]], %[[INSERT0]][%[[C1_0]] : i32] : vector<3xf32>
  // CHECK-NEXT: llvm.return %[[RES]] : vector<3xf32>
  %0 = spv.VectorShuffle [0: i32, 4: i32, 0xffffffff: i32] %vector1: vector<3xf32>, %vector2: vector<2xf32> -> vector<3xf32>
  spv.ReturnValue %0: vector<3xf32>
}

//===----------------------------------------------------------------------===//
// spv.EntryPoint and spv.ExecutionMode
//===----------------------------------------------------------------------===//

//      CHECK: module {
// CHECK-NEXT:   llvm.mlir.global external constant @{{.*}}() : !llvm.struct<(i32)> {
// CHECK-NEXT:     %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[VAL:.*]] = llvm.mlir.constant(31 : i32) : i32
// CHECK-NEXT:     %[[RET:.*]] = llvm.insertvalue %[[VAL]], %[[UNDEF]][0] : !llvm.struct<(i32)>
// CHECK-NEXT:     llvm.return %[[RET]] : !llvm.struct<(i32)>
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func @empty
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
spv.module Logical OpenCL {
  spv.func @empty() "None" {
    spv.Return
  }
  spv.EntryPoint "Kernel" @empty
  spv.ExecutionMode @empty "ContractionOff"
}

//      CHECK: module {
// CHECK-NEXT:   llvm.mlir.global external constant @{{.*}}() : !llvm.struct<(i32, array<3 x i32>)> {
// CHECK-NEXT:     %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, array<3 x i32>)>
// CHECK-NEXT:     %[[EM:.*]] = llvm.mlir.constant(18 : i32) : i32
// CHECK-NEXT:     %[[T0:.*]] = llvm.insertvalue %[[EM]], %[[UNDEF]][0] : !llvm.struct<(i32, array<3 x i32>)>
// CHECK-NEXT:     %[[C0:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK-NEXT:     %[[T1:.*]] = llvm.insertvalue %[[C0]], %[[T0]][1, 0] : !llvm.struct<(i32, array<3 x i32>)>
// CHECK-NEXT:     %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %[[T2:.*]] = llvm.insertvalue %[[C1]], %[[T1]][1, 1] : !llvm.struct<(i32, array<3 x i32>)>
// CHECK-NEXT:     %[[C2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:     %[[RET:.*]] = llvm.insertvalue %[[C2]], %[[T2]][1, 2] : !llvm.struct<(i32, array<3 x i32>)>
// CHECK-NEXT:     llvm.return %[[RET]] : !llvm.struct<(i32, array<3 x i32>)>
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.mlir.global external constant @{{.*}}() : !llvm.struct<(i32)> {
//      CHECK:   llvm.func @bar
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
spv.module Logical OpenCL {
  spv.func @bar() "None" {
    spv.Return
  }
  spv.EntryPoint "Kernel" @bar
  spv.ExecutionMode @bar "ContractionOff"
  spv.ExecutionMode @bar "LocalSizeHint", 32, 1, 1
}

//===----------------------------------------------------------------------===//
// spv.Undef
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @undef_scalar
spv.func @undef_scalar() "None" {
  // CHECK: llvm.mlir.undef : f32
  %0 = spv.Undef : f32
  spv.Return
}

// CHECK-LABEL: @undef_vector
spv.func @undef_vector() "None" {
  // CHECK: llvm.mlir.undef : vector<2xi32>
  %0 = spv.Undef : vector<2xi32>
  spv.Return
}
