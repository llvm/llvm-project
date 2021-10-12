// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #map0 = affine_map<(d0) -> (d0 + 1)>

// CHECK: #map1 = affine_map<()[s0] -> (s0 + 1)>

// CHECK-DAG: #[[$BASE_MAP0:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>
// CHECK-DAG: #[[$BASE_MAP3:map[0-9]+]] = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>

// CHECK-DAG: #[[$BASE_MAP1:map[0-9]+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[$SUBVIEW_MAP1:map[0-9]+]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-DAG: #[[$BASE_MAP2:map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 22 + d1)>
// CHECK-DAG: #[[$SUBVIEW_MAP2:map[0-9]+]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
// CHECK-DAG: #[[$SUBVIEW_MAP3:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>
// CHECK-DAG: #[[$SUBVIEW_MAP4:map[0-9]+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$SUBVIEW_MAP5:map[0-9]+]] = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1 * 2)>
// CHECK-DAG: #[[$SUBVIEW_MAP6:map[0-9]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 36 + d1 * 36 + d2 * 4 + d3 * 4 + d4)>
// CHECK-DAG: #[[$SUBVIEW_MAP7:map[0-9]+]] = affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4 + d4 * s5 + d5 * s6)>
// CHECK-DAG: #[[$SUBVIEW_MAP8:map[0-9]+]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>
// CHECK-DAG: #[[$SUBVIEW_MAP9:map[0-9]+]] = affine_map<(d0, d1) -> (d0 * 3 + d1 + 6)>
// CHECK-DAG: #[[$SUBVIEW_MAP10:map[0-9]+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG: #[[$SUBVIEW_MAP11:map[0-9]+]] = affine_map<() -> (4)>
// CHECK-DAG: #[[$SUBVIEW_MAP12:map[0-9]+]] = affine_map<()[s0] -> (s0)>

// CHECK-LABEL: func @func_with_ops
// CHECK-SAME: %[[ARG:.*]]: f32
func @func_with_ops(f32) {
^bb0(%a : f32):
  // CHECK: %[[T:.*]] = "getTensor"() : () -> tensor<4x4x?xf32>
  %t = "getTensor"() : () -> tensor<4x4x?xf32>

  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %{{.*}} = tensor.dim %[[T]], %[[C2]] : tensor<4x4x?xf32>
  %c2 = arith.constant 2 : index
  %t2 = "tensor.dim"(%t, %c2) : (tensor<4x4x?xf32>, index) -> index

  // CHECK: %{{.*}} = arith.addf %[[ARG]], %[[ARG]] : f32
  %x = "arith.addf"(%a, %a) : (f32,f32) -> (f32)

  // CHECK: return
  return
}

// CHECK-LABEL: func @standard_instrs(%arg0: tensor<4x4x?xf32>, %arg1: f32, %arg2: i32, %arg3: index, %arg4: i64, %arg5: f16) {
func @standard_instrs(tensor<4x4x?xf32>, f32, i32, index, i64, f16) {
^bb42(%t: tensor<4x4x?xf32>, %f: f32, %i: i32, %idx : index, %j: i64, %half: f16):
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[A2:.*]] = tensor.dim %arg0, %[[C2]] : tensor<4x4x?xf32>
  %c2 = arith.constant 2 : index
  %a2 = tensor.dim %t, %c2 : tensor<4x4x?xf32>

  // CHECK: %f = constant @func_with_ops : (f32) -> ()
  %10 = constant @func_with_ops : (f32) -> ()

  // CHECK: %f_0 = constant @affine_apply : () -> ()
  %11 = constant @affine_apply : () -> ()

  // CHECK: %[[I2:.*]] = arith.addi
  %i2 = arith.addi %i, %i: i32
  // CHECK: %[[I3:.*]] = arith.addi
  %i3 = arith.addi %i2, %i : i32
  // CHECK: %[[I4:.*]] = arith.addi
  %i4 = arith.addi %i2, %i3 : i32
  // CHECK: %[[F3:.*]] = arith.addf
  %f3 = arith.addf %f, %f : f32
  // CHECK: %[[F4:.*]] = arith.addf
  %f4 = arith.addf %f, %f3 : f32

  %true = arith.constant true
  %tci32 = arith.constant dense<0> : tensor<42xi32>
  %vci32 = arith.constant dense<0> : vector<42xi32>
  %tci1 = arith.constant dense<1> : tensor<42xi1>
  %vci1 = arith.constant dense<1> : vector<42xi1>

  // CHECK: %{{.*}} = select %{{.*}}, %arg3, %arg3 : index
  %21 = select %true, %idx, %idx : index

  // CHECK: %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : tensor<42xi1>, tensor<42xi32>
  %22 = select %tci1, %tci32, %tci32 : tensor<42 x i1>, tensor<42 x i32>

  // CHECK: %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : vector<42xi1>, vector<42xi32>
  %23 = select %vci1, %vci32, %vci32 : vector<42 x i1>, vector<42 x i32>

  // CHECK: %{{.*}} = select %{{.*}}, %arg3, %arg3 : index
  %24 = "std.select"(%true, %idx, %idx) : (i1, index, index) -> index

  // CHECK: %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : tensor<42xi32>
  %25 = std.select %true, %tci32, %tci32 : tensor<42 x i32>

  %64 = arith.constant dense<0.> : vector<4 x f32>
  %tcf32 = arith.constant dense<0.> : tensor<42 x f32>
  %vcf32 = arith.constant dense<0.> : vector<4 x f32>

  // CHECK: %{{.*}} = arith.cmpf ogt, %{{.*}}, %{{.*}} : f32
  %65 = arith.cmpf ogt, %f3, %f4 : f32

  // Predicate 0 means ordered equality comparison.
  // CHECK: %{{.*}} = arith.cmpf oeq, %{{.*}}, %{{.*}} : f32
  %66 = "arith.cmpf"(%f3, %f4) {predicate = 1} : (f32, f32) -> i1

  // CHECK: %{{.*}} = arith.cmpf olt, %{{.*}}, %{{.*}}: vector<4xf32>
  %67 = arith.cmpf olt, %vcf32, %vcf32 : vector<4 x f32>

  // CHECK: %{{.*}} = arith.cmpf oeq, %{{.*}}, %{{.*}}: vector<4xf32>
  %68 = "arith.cmpf"(%vcf32, %vcf32) {predicate = 1} : (vector<4 x f32>, vector<4 x f32>) -> vector<4 x i1>

  // CHECK: %{{.*}} = arith.cmpf oeq, %{{.*}}, %{{.*}}: tensor<42xf32>
  %69 = arith.cmpf oeq, %tcf32, %tcf32 : tensor<42 x f32>

  // CHECK: %{{.*}} = arith.cmpf oeq, %{{.*}}, %{{.*}}: vector<4xf32>
  %70 = arith.cmpf oeq, %vcf32, %vcf32 : vector<4 x f32>

  // CHECK: %{{.*}} = rank %arg0 : tensor<4x4x?xf32>
  %71 = "std.rank"(%t) : (tensor<4x4x?xf32>) -> index

  // CHECK: %{{.*}} = rank %arg0 : tensor<4x4x?xf32>
  %72 = rank %t : tensor<4x4x?xf32>

  // CHECK: = constant unit
  %73 = constant unit

  // CHECK: arith.constant true
  %74 = arith.constant true

  // CHECK: arith.constant false
  %75 = arith.constant false

  // CHECK: %{{.*}} = math.abs %arg1 : f32
  %100 = "math.abs"(%f) : (f32) -> f32

  // CHECK: %{{.*}} = math.abs %arg1 : f32
  %101 = math.abs %f : f32

  // CHECK: %{{.*}} = math.abs %{{.*}}: vector<4xf32>
  %102 = math.abs %vcf32 : vector<4xf32>

  // CHECK: %{{.*}} = math.abs %arg0 : tensor<4x4x?xf32>
  %103 = math.abs %t : tensor<4x4x?xf32>

  // CHECK: %{{.*}} = math.ceil %arg1 : f32
  %104 = "math.ceil"(%f) : (f32) -> f32

  // CHECK: %{{.*}} = math.ceil %arg1 : f32
  %105 = math.ceil %f : f32

  // CHECK: %{{.*}} = math.ceil %{{.*}}: vector<4xf32>
  %106 = math.ceil %vcf32 : vector<4xf32>

  // CHECK: %{{.*}} = math.ceil %arg0 : tensor<4x4x?xf32>
  %107 = math.ceil %t : tensor<4x4x?xf32>

  // CHECK: %{{.*}} = math.copysign %arg1, %arg1 : f32
  %116 = "math.copysign"(%f, %f) : (f32, f32) -> f32

  // CHECK: %{{.*}} = math.copysign %arg1, %arg1 : f32
  %117 = math.copysign %f, %f : f32

  // CHECK: %{{.*}} = math.copysign %{{.*}}, %{{.*}}: vector<4xf32>
  %118 = math.copysign %vcf32, %vcf32 : vector<4xf32>

  // CHECK: %{{.*}} = math.copysign %arg0, %arg0 : tensor<4x4x?xf32>
  %119 = math.copysign %t, %t : tensor<4x4x?xf32>

  // CHECK: %{{.*}} = math.rsqrt %arg1 : f32
  %145 = math.rsqrt %f : f32

  // CHECK: math.floor %arg1 : f32
  %163 = "math.floor"(%f) : (f32) -> f32

  // CHECK: %{{.*}} = math.floor %arg1 : f32
  %164 = math.floor %f : f32

  // CHECK: %{{.*}} = math.floor %{{.*}}: vector<4xf32>
  %165 = math.floor %vcf32 : vector<4xf32>

  // CHECK: %{{.*}} = math.floor %arg0 : tensor<4x4x?xf32>
  %166 = math.floor %t : tensor<4x4x?xf32>

  return
}

// CHECK-LABEL: func @affine_apply() {
func @affine_apply() {
  %i = "arith.constant"() {value = 0: index} : () -> index
  %j = "arith.constant"() {value = 1: index} : () -> index

  // CHECK: affine.apply #map0(%c0)
  %a = "affine.apply" (%i) { map = affine_map<(d0) -> (d0 + 1)> } :
    (index) -> (index)

  // CHECK: affine.apply #map1()[%c0]
  %b = affine.apply affine_map<()[x] -> (x+1)>()[%i]

  return
}

// CHECK-LABEL: func @load_store_prefetch
func @load_store_prefetch(memref<4x4xi32>, index) {
^bb0(%0: memref<4x4xi32>, %1: index):
  // CHECK: %0 = memref.load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %2 = "memref.load"(%0, %1, %1) : (memref<4x4xi32>, index, index)->i32

  // CHECK: %{{.*}} = memref.load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %3 = memref.load %0[%1, %1] : memref<4x4xi32>

  // CHECK: memref.prefetch %arg0[%arg1, %arg1], write, locality<1>, data : memref<4x4xi32>
  memref.prefetch %0[%1, %1], write, locality<1>, data : memref<4x4xi32>

  // CHECK: memref.prefetch %arg0[%arg1, %arg1], read, locality<3>, instr : memref<4x4xi32>
  memref.prefetch %0[%1, %1], read, locality<3>, instr : memref<4x4xi32>

  return
}

// Test with zero-dimensional operands using no index in load/store.
// CHECK-LABEL: func @zero_dim_no_idx
func @zero_dim_no_idx(%arg0 : memref<i32>, %arg1 : memref<i32>, %arg2 : memref<i32>) {
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  return
  // CHECK: %0 = memref.load %{{.*}}[] : memref<i32>
  // CHECK: memref.store %{{.*}}, %{{.*}}[] : memref<i32>
}

// CHECK-LABEL: func @return_op(%arg0: i32) -> i32 {
func @return_op(%a : i32) -> i32 {
  // CHECK: return %arg0 : i32
  "std.return" (%a) : (i32)->()
}

// CHECK-LABEL: func @calls(%arg0: i32) {
func @calls(%arg0: i32) {
  // CHECK: %0 = call @return_op(%arg0) : (i32) -> i32
  %x = call @return_op(%arg0) : (i32) -> i32
  // CHECK: %1 = call @return_op(%0) : (i32) -> i32
  %y = call @return_op(%x) : (i32) -> i32
  // CHECK: %2 = call @return_op(%0) : (i32) -> i32
  %z = "std.call"(%x) {callee = @return_op} : (i32) -> i32

  // CHECK: %f = constant @affine_apply : () -> ()
  %f = constant @affine_apply : () -> ()

  // CHECK: call_indirect %f() : () -> ()
  call_indirect %f() : () -> ()

  // CHECK: %f_0 = constant @return_op : (i32) -> i32
  %f_0 = constant @return_op : (i32) -> i32

  // CHECK: %3 = call_indirect %f_0(%arg0) : (i32) -> i32
  %2 = call_indirect %f_0(%arg0) : (i32) -> i32

  // CHECK: %4 = call_indirect %f_0(%arg0) : (i32) -> i32
  %3 = "std.call_indirect"(%f_0, %arg0) : ((i32) -> i32, i32) -> i32

  return
}

// CHECK-LABEL: func @memref_cast(%arg0
func @memref_cast(%arg0: memref<4xf32>, %arg1 : memref<?xf32>, %arg2 : memref<64x16x4xf32, offset: 0, strides: [64, 4, 1]>) {
  // CHECK: %0 = memref.cast %arg0 : memref<4xf32> to memref<?xf32>
  %0 = memref.cast %arg0 : memref<4xf32> to memref<?xf32>

  // CHECK: %1 = memref.cast %arg1 : memref<?xf32> to memref<4xf32>
  %1 = memref.cast %arg1 : memref<?xf32> to memref<4xf32>

  // CHECK: {{%.*}} = memref.cast %arg2 : memref<64x16x4xf32, #[[$BASE_MAP0]]> to memref<64x16x4xf32, #[[$BASE_MAP3]]>
  %2 = memref.cast %arg2 : memref<64x16x4xf32, offset: 0, strides: [64, 4, 1]> to memref<64x16x4xf32, offset: ?, strides: [?, ?, ?]>

  // CHECK: {{%.*}} = memref.cast {{%.*}} : memref<64x16x4xf32, #[[$BASE_MAP3]]> to memref<64x16x4xf32, #[[$BASE_MAP0]]>
  %3 = memref.cast %2 : memref<64x16x4xf32, offset: ?, strides: [?, ?, ?]> to memref<64x16x4xf32, offset: 0, strides: [64, 4, 1]>

  // CHECK: memref.cast %{{.*}} : memref<4xf32> to memref<*xf32>
  %4 = memref.cast %1 : memref<4xf32> to memref<*xf32>

  // CHECK: memref.cast %{{.*}} : memref<*xf32> to memref<4xf32>
  %5 = memref.cast %4 : memref<*xf32> to memref<4xf32>
  return
}

// Check that unranked memrefs with non-default memory space roundtrip
// properly.
// CHECK-LABEL: @unranked_memref_roundtrip(memref<*xf32, 4>)
func private @unranked_memref_roundtrip(memref<*xf32, 4>)

// CHECK-LABEL: func @memref_view(%arg0
func @memref_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8>
  // Test two dynamic sizes and dynamic offset.
  // CHECK: %{{.*}} = memref.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>
  %1 = memref.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>

  // Test one dynamic size and dynamic offset.
  // CHECK: %{{.*}} = memref.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>
  %3 = memref.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>

  // Test static sizes and static offset.
  // CHECK: %{{.*}} = memref.view %0[{{.*}}][] : memref<2048xi8> to memref<64x4xf32>
  %c0 = arith.constant 0: index
  %5 = memref.view %0[%c0][] : memref<2048xi8> to memref<64x4xf32>
  return
}

// CHECK-LABEL: func @memref_subview(%arg0
func @memref_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = memref.alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>
  // CHECK: subview %0[%c0, %c0, %c0] [%arg0, %arg1, %arg2] [%c1, %c1, %c1] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]>
  // CHECK-SAME: to memref<?x?x?xf32, #[[$BASE_MAP3]]>
  %1 = memref.subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1]
    : memref<8x16x4xf32, offset:0, strides: [64, 4, 1]> to
      memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>

  %2 = memref.alloc()[%arg2] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  // CHECK: memref.subview %2[%c1] [%arg0] [%c1] :
  // CHECK-SAME: memref<64xf32, #[[$BASE_MAP1]]>
  // CHECK-SAME: to memref<?xf32, #[[$SUBVIEW_MAP1]]>
  %3 = memref.subview %2[%c1][%arg0][%c1]
    : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to
      memref<?xf32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>

  %4 = memref.alloc() : memref<64x22xf32, affine_map<(d0, d1) -> (d0 * 22 + d1)>>
  // CHECK: memref.subview %4[%c0, %c1] [%arg0, %arg1] [%c1, %c0] :
  // CHECK-SAME: memref<64x22xf32, #[[$BASE_MAP2]]>
  // CHECK-SAME: to memref<?x?xf32, #[[$SUBVIEW_MAP2]]>
  %5 = memref.subview %4[%c0, %c1][%arg0, %arg1][%c1, %c0]
    : memref<64x22xf32, offset:0, strides: [22, 1]> to
      memref<?x?xf32, offset:?, strides: [?, ?]>

  // CHECK: memref.subview %0[0, 2, 0] [4, 4, 4] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]>
  // CHECK-SAME: to memref<4x4x4xf32, #[[$SUBVIEW_MAP3]]>
  %6 = memref.subview %0[0, 2, 0][4, 4, 4][1, 1, 1]
    : memref<8x16x4xf32, offset:0, strides: [64, 4, 1]> to
      memref<4x4x4xf32, offset:8, strides: [64, 4, 1]>

  %7 = memref.alloc(%arg1, %arg2) : memref<?x?xf32>
  // CHECK: memref.subview {{%.*}}[0, 0] [4, 4] [1, 1] :
  // CHECK-SAME: memref<?x?xf32>
  // CHECK-SAME: to memref<4x4xf32, #[[$SUBVIEW_MAP4]]>
  %8 = memref.subview %7[0, 0][4, 4][1, 1]
    : memref<?x?xf32> to memref<4x4xf32, offset: ?, strides:[?, 1]>

  %9 = memref.alloc() : memref<16x4xf32>
  // CHECK: memref.subview {{%.*}}[{{%.*}}, {{%.*}}] [4, 4] [{{%.*}}, {{%.*}}] :
  // CHECK-SAME: memref<16x4xf32>
  // CHECK-SAME: to memref<4x4xf32, #[[$SUBVIEW_MAP2]]
  %10 = memref.subview %9[%arg1, %arg1][4, 4][%arg2, %arg2]
    : memref<16x4xf32> to memref<4x4xf32, offset: ?, strides:[?, ?]>

  // CHECK: memref.subview {{%.*}}[{{%.*}}, {{%.*}}] [4, 4] [2, 2] :
  // CHECK-SAME: memref<16x4xf32>
  // CHECK-SAME: to memref<4x4xf32, #[[$SUBVIEW_MAP5]]
  %11 = memref.subview %9[%arg1, %arg2][4, 4][2, 2]
    : memref<16x4xf32> to memref<4x4xf32, offset: ?, strides:[8, 2]>

  %12 = memref.alloc() : memref<1x9x1x4x1xf32, affine_map<(d0, d1, d2, d3, d4) -> (36 * d0 + 36 * d1 + 4 * d2 + 4 * d3 + d4)>>
  // CHECK: memref.subview %12[%arg1, %arg1, %arg1, %arg1, %arg1]
  // CHECK-SAME: [1, 9, 1, 4, 1] [%arg2, %arg2, %arg2, %arg2, %arg2] :
  // CHECK-SAME: memref<1x9x1x4x1xf32, #[[$SUBVIEW_MAP6]]> to memref<9x4xf32, #[[$SUBVIEW_MAP2]]>
  %13 = memref.subview %12[%arg1, %arg1, %arg1, %arg1, %arg1][1, 9, 1, 4, 1][%arg2, %arg2, %arg2, %arg2, %arg2] : memref<1x9x1x4x1xf32, offset: 0, strides: [36, 36, 4, 4, 1]> to memref<9x4xf32, offset: ?, strides: [?, ?]>
  // CHECK: memref.subview %12[%arg1, %arg1, %arg1, %arg1, %arg1]
  // CHECK-SAME: [1, 9, 1, 4, 1] [%arg2, %arg2, %arg2, %arg2, %arg2] :
  // CHECK-SAME: memref<1x9x1x4x1xf32, #[[$SUBVIEW_MAP6]]> to memref<1x9x4xf32, #[[$BASE_MAP3]]>
  %14 = memref.subview %12[%arg1, %arg1, %arg1, %arg1, %arg1][1, 9, 1, 4, 1][%arg2, %arg2, %arg2, %arg2, %arg2] : memref<1x9x1x4x1xf32, offset: 0, strides: [36, 36, 4, 4, 1]> to memref<1x9x4xf32, offset: ?, strides: [?, ?, ?]>

  %15 = memref.alloc(%arg1, %arg2)[%c0, %c1, %arg1, %arg0, %arg0, %arg2, %arg2] : memref<1x?x5x1x?x1xf32, affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6] -> (s0 + s1 * d0 + s2 * d1 + s3 * d2 + s4 * d3 + s5 * d4 + s6 * d5)>>
  // CHECK: memref.subview %15[0, 0, 0, 0, 0, 0] [1, %arg1, 5, 1, %arg2, 1] [1, 1, 1, 1, 1, 1]  :
  // CHECK-SAME: memref<1x?x5x1x?x1xf32,  #[[$SUBVIEW_MAP7]]> to memref<?x5x?xf32, #[[$BASE_MAP3]]>
  %16 = memref.subview %15[0, 0, 0, 0, 0, 0][1, %arg1, 5, 1, %arg2, 1][1, 1, 1, 1, 1, 1] : memref<1x?x5x1x?x1xf32, offset: ?, strides: [?, ?, ?, ?, ?, ?]> to memref<?x5x?xf32, offset: ?, strides: [?, ?, ?]>
  // CHECK: memref.subview %15[%arg1, %arg1, %arg1, %arg1, %arg1, %arg1] [1, %arg1, 5, 1, %arg2, 1] [1, 1, 1, 1, 1, 1]  :
  // CHECK-SAME: memref<1x?x5x1x?x1xf32, #[[$SUBVIEW_MAP7]]> to memref<?x5x?x1xf32, #[[$SUBVIEW_MAP8]]>
  %17 = memref.subview %15[%arg1, %arg1, %arg1, %arg1, %arg1, %arg1][1, %arg1, 5, 1, %arg2, 1][1, 1, 1, 1, 1, 1] :  memref<1x?x5x1x?x1xf32, offset: ?, strides: [?, ?, ?, ?, ?, ?]> to memref<?x5x?x1xf32, offset: ?, strides: [?, ?, ?, ?]>

  %18 = memref.alloc() : memref<1x8xf32>
  // CHECK: memref.subview %18[0, 0] [1, 8] [1, 1]  : memref<1x8xf32> to memref<8xf32>
  %19 = memref.subview %18[0, 0][1, 8][1, 1] : memref<1x8xf32> to memref<8xf32>

  %20 = memref.alloc() : memref<8x16x4xf32>
  // CHECK: memref.subview %20[0, 0, 0] [1, 16, 4] [1, 1, 1]  : memref<8x16x4xf32> to memref<16x4xf32>
  %21 = memref.subview %20[0, 0, 0][1, 16, 4][1, 1, 1] : memref<8x16x4xf32> to memref<16x4xf32>

  %22 = memref.subview %20[3, 4, 2][1, 6, 3][1, 1, 1] : memref<8x16x4xf32> to memref<6x3xf32, offset: 210, strides: [4, 1]>

  %23 = memref.alloc() : memref<f32>
  %78 = memref.subview %23[] [] []  : memref<f32> to memref<f32>

  /// Subview with only leading operands.
  %24 = memref.alloc() : memref<5x3xf32>
  // CHECK: memref.subview %{{.*}}[2] [3] [1] : memref<5x3xf32> to memref<3x3xf32, #[[$SUBVIEW_MAP9]]>
  %25 = memref.subview %24[2][3][1]: memref<5x3xf32> to memref<3x3xf32, offset: 6, strides: [3, 1]>

  /// Rank-reducing subview with only leading operands.
  // CHECK: memref.subview %{{.*}}[1] [1] [1] : memref<5x3xf32> to memref<3xf32, #[[$SUBVIEW_MAP10]]>
  %26 = memref.subview %24[1][1][1]: memref<5x3xf32> to memref<3xf32, offset: 3, strides: [1]>

  // Corner-case of 0-D rank-reducing subview with an offset.
  // CHECK: memref.subview %{{.*}}[1, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, #[[$SUBVIEW_MAP11]]>
  %27 = memref.subview %24[1, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, affine_map<() -> (4)>>

  // CHECK: memref.subview %{{.*}}[%{{.*}}, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, #[[$SUBVIEW_MAP12]]>
  %28 = memref.subview %24[%arg0, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, affine_map<()[s0] -> (s0)>>

  // CHECK: memref.subview %{{.*}}[0, %{{.*}}] [%{{.*}}, 1] [1, 1] : memref<?x?xf32> to memref<?xf32, #[[$SUBVIEW_MAP1]]>
  %a30 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %30 = memref.subview %a30[0, %arg1][%arg2, 1][1, 1] : memref<?x?xf32> to memref<?xf32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>

  return
}

// CHECK-LABEL: func @test_dimop
// CHECK-SAME: %[[ARG:.*]]: tensor<4x4x?xf32>
func @test_dimop(%arg0: tensor<4x4x?xf32>) {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %{{.*}} = tensor.dim %[[ARG]], %[[C2]] : tensor<4x4x?xf32>
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c2 : tensor<4x4x?xf32>
  // use dim as an index to ensure type correctness
  %1 = affine.apply affine_map<(d0) -> (d0)>(%0)
  return
}

// CHECK-LABEL: func @test_splat_op
// CHECK-SAME: [[S:%arg[0-9]+]]: f32
func @test_splat_op(%s : f32) {
  %v = splat %s : vector<8xf32>
  // CHECK: splat [[S]] : vector<8xf32>
  %t = splat %s : tensor<8xf32>
  // CHECK: splat [[S]] : tensor<8xf32>
  %u = "std.splat"(%s) : (f32) -> vector<4xf32>
  // CHECK: splat [[S]] : vector<4xf32>
  return
}

// CHECK-LABEL: func @tensor_load_store
func @tensor_load_store(%0 : memref<4x4xi32>) {
  // CHECK: %[[TENSOR:.*]] = memref.tensor_load %[[MEMREF:.*]] : memref<4x4xi32>
  %1 = memref.tensor_load %0 : memref<4x4xi32>
  // CHECK: memref.tensor_store %[[TENSOR]], %[[MEMREF]] : memref<4x4xi32>
  memref.tensor_store %1, %0 : memref<4x4xi32>
  return
}

// CHECK-LABEL: func @unranked_tensor_load_store
func @unranked_tensor_load_store(%0 : memref<*xi32>) {
  // CHECK: %[[TENSOR:.*]] = memref.tensor_load %[[MEMREF:.*]] : memref<*xi32>
  %1 = memref.tensor_load %0 : memref<*xi32>
  // CHECK: memref.tensor_store %[[TENSOR]], %[[MEMREF]] : memref<*xi32>
  memref.tensor_store %1, %0 : memref<*xi32>
  return
}

// CHECK-LABEL: func @atomic_rmw
// CHECK-SAME: ([[BUF:%.*]]: memref<10xf32>, [[VAL:%.*]]: f32, [[I:%.*]]: index)
func @atomic_rmw(%I: memref<10xf32>, %val: f32, %i : index) {
  %x = atomic_rmw addf %val, %I[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: atomic_rmw addf [[VAL]], [[BUF]]{{\[}}[[I]]]
  return
}

// CHECK-LABEL: func @generic_atomic_rmw
// CHECK-SAME: ([[BUF:%.*]]: memref<1x2xf32>, [[I:%.*]]: index, [[J:%.*]]: index)
func @generic_atomic_rmw(%I: memref<1x2xf32>, %i : index, %j : index) {
  %x = generic_atomic_rmw %I[%i, %j] : memref<1x2xf32> {
  // CHECK-NEXT: generic_atomic_rmw [[BUF]]{{\[}}[[I]], [[J]]] : memref
    ^bb0(%old_value : f32):
      %c1 = arith.constant 1.0 : f32
      %out = arith.addf %c1, %old_value : f32
      atomic_yield %out : f32
  // CHECK: index_attr = 8 : index
  } { index_attr = 8 : index }
  return
}

// CHECK-LABEL: func @assume_alignment
// CHECK-SAME: %[[MEMREF:.*]]: memref<4x4xf16>
func @assume_alignment(%0: memref<4x4xf16>) {
  // CHECK: memref.assume_alignment %[[MEMREF]], 16 : memref<4x4xf16>
  memref.assume_alignment %0, 16 : memref<4x4xf16>
  return
}

// CHECK-LABEL: func @slice({{.*}}) {
func @slice(%t: tensor<8x16x4xf32>, %idx : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<?x?x?xf32>
  %1 = tensor.extract_slice %t[%c0, %c0, %c0][%idx, %idx, %idx][%c1, %c1, %c1]
    : tensor<8x16x4xf32> to tensor<?x?x?xf32>

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<4x4x4xf32>
  %2 = tensor.extract_slice %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<4x4xf32>
  %3 = tensor.extract_slice %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4xf32>

  return
}

// CHECK-LABEL: func @insert_slice({{.*}}) {
func @insert_slice(
    %t: tensor<8x16x4xf32>,
    %t2: tensor<16x32x8xf32>,
    %t3: tensor<4x4xf32>,
    %idx : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<8x16x4xf32> into tensor<16x32x8xf32>
  %1 = tensor.insert_slice %t into %t2[%c0, %c0, %c0][%idx, %idx, %idx][%c1, %c1, %c1]
    : tensor<8x16x4xf32> into tensor<16x32x8xf32>

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<8x16x4xf32> into tensor<16x32x8xf32>
  %2 = tensor.insert_slice %t into %t2[%c0, %idx, %c0][%idx, 4, %idx][%c1, 1, %c1]
    : tensor<8x16x4xf32> into tensor<16x32x8xf32>

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<4x4xf32> into tensor<8x16x4xf32>
  %3 = tensor.insert_slice %t3 into %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : tensor<4x4xf32> into tensor<8x16x4xf32>

  return
}
