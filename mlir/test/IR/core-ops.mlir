// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0 + 1)>

// CHECK: #map1 = affine_map<()[s0] -> (s0 + 1)>

// CHECK-LABEL: func @func_with_ops
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @func_with_ops(f32) {
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
func.func @standard_instrs(tensor<4x4x?xf32>, f32, i32, index, i64, f16) {
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

  // CHECK: %{{.*}} = arith.select %{{.*}}, %arg3, %arg3 : index
  %21 = arith.select %true, %idx, %idx : index

  // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<42xi1>, tensor<42xi32>
  %22 = arith.select %tci1, %tci32, %tci32 : tensor<42 x i1>, tensor<42 x i32>

  // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<42xi1>, vector<42xi32>
  %23 = arith.select %vci1, %vci32, %vci32 : vector<42 x i1>, vector<42 x i32>

  // CHECK: %{{.*}} = arith.select %{{.*}}, %arg3, %arg3 : index
  %24 = "arith.select"(%true, %idx, %idx) : (i1, index, index) -> index

  // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<42xi32>
  %25 = arith.select %true, %tci32, %tci32 : tensor<42 x i32>

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

  // CHECK: arith.constant true
  %74 = arith.constant true

  // CHECK: arith.constant false
  %75 = arith.constant false

  // CHECK: %{{.*}} = math.absf %arg1 : f32
  %100 = "math.absf"(%f) : (f32) -> f32

  // CHECK: %{{.*}} = math.absf %arg1 : f32
  %101 = math.absf %f : f32

  // CHECK: %{{.*}} = math.absf %{{.*}}: vector<4xf32>
  %102 = math.absf %vcf32 : vector<4xf32>

  // CHECK: %{{.*}} = math.absf %arg0 : tensor<4x4x?xf32>
  %103 = math.absf %t : tensor<4x4x?xf32>

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
func.func @affine_apply() {
  %i = "arith.constant"() {value = 0: index} : () -> index
  %j = "arith.constant"() {value = 1: index} : () -> index

  // CHECK: affine.apply #map(%c0)
  %a = "affine.apply" (%i) { map = affine_map<(d0) -> (d0 + 1)> } :
    (index) -> (index)

  // CHECK: affine.apply #map1()[%c0]
  %b = affine.apply affine_map<()[x] -> (x+1)>()[%i]

  return
}

// CHECK-LABEL: func @return_op(%arg0: i32) -> i32 {
func.func @return_op(%a : i32) -> i32 {
  // CHECK: return %arg0 : i32
  "func.return" (%a) : (i32)->()
}

// CHECK-LABEL: func @calls(%arg0: i32) {
func.func @calls(%arg0: i32) {
  // CHECK: %0 = call @return_op(%arg0) : (i32) -> i32
  %x = call @return_op(%arg0) : (i32) -> i32
  // CHECK: %1 = call @return_op(%0) : (i32) -> i32
  %y = call @return_op(%x) : (i32) -> i32
  // CHECK: %2 = call @return_op(%0) : (i32) -> i32
  %z = "func.call"(%x) {callee = @return_op} : (i32) -> i32

  // CHECK: %f = constant @affine_apply : () -> ()
  %f = constant @affine_apply : () -> ()

  // CHECK: call_indirect %f() : () -> ()
  call_indirect %f() : () -> ()

  // CHECK: %f_0 = constant @return_op : (i32) -> i32
  %f_0 = constant @return_op : (i32) -> i32

  // CHECK: %3 = call_indirect %f_0(%arg0) : (i32) -> i32
  %2 = call_indirect %f_0(%arg0) : (i32) -> i32

  // CHECK: %4 = call_indirect %f_0(%arg0) : (i32) -> i32
  %3 = "func.call_indirect"(%f_0, %arg0) : ((i32) -> i32, i32) -> i32

  return
}

// CHECK-LABEL: func @test_dimop
// CHECK-SAME: %[[ARG:.*]]: tensor<4x4x?xf32>
func.func @test_dimop(%arg0: tensor<4x4x?xf32>) {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %{{.*}} = tensor.dim %[[ARG]], %[[C2]] : tensor<4x4x?xf32>
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c2 : tensor<4x4x?xf32>
  // use dim as an index to ensure type correctness
  %1 = affine.apply affine_map<(d0) -> (d0)>(%0)
  return
}
