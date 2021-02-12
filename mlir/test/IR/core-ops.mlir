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

  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK-NEXT: %{{.*}} = dim %[[T]], %[[C2]] : tensor<4x4x?xf32>
  %c2 = constant 2 : index
  %t2 = "std.dim"(%t, %c2) : (tensor<4x4x?xf32>, index) -> index

  // CHECK: %{{.*}} = addf %[[ARG]], %[[ARG]] : f32
  %x = "std.addf"(%a, %a) : (f32,f32) -> (f32)

  // CHECK: return
  return
}

// CHECK-LABEL: func @standard_instrs(%arg0: tensor<4x4x?xf32>, %arg1: f32, %arg2: i32, %arg3: index, %arg4: i64, %arg5: f16) {
func @standard_instrs(tensor<4x4x?xf32>, f32, i32, index, i64, f16) {
^bb42(%t: tensor<4x4x?xf32>, %f: f32, %i: i32, %idx : index, %j: i64, %half: f16):
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: %[[A2:.*]] = dim %arg0, %[[C2]] : tensor<4x4x?xf32>
  %c2 = constant 2 : index
  %a2 = dim %t, %c2 : tensor<4x4x?xf32>

  // CHECK: %[[F2:.*]] = addf %arg1, %arg1 : f32
  %f2 = "std.addf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %[[F3:.*]] = addf %[[F2]], %[[F2]] : f32
  %f3 = addf %f2, %f2 : f32

  // CHECK: %[[I2:.*]] = addi %arg2, %arg2 : i32
  %i2 = "std.addi"(%i, %i) : (i32,i32) -> i32

  // CHECK: %[[I3:.*]] = addi %[[I2]], %[[I2]] : i32
  %i3 = addi %i2, %i2 : i32

  // CHECK: %[[IDX1:.*]] = addi %arg3, %arg3 : index
  %idx1 = addi %idx, %idx : index

  // CHECK: %[[IDX2:.*]] = addi %arg3, %[[IDX1]] : index
  %idx2 = "std.addi"(%idx, %idx1) : (index, index) -> index

  // CHECK: %[[F4:.*]] = subf %arg1, %arg1 : f32
  %f4 = "std.subf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %[[F5:.*]] = subf %[[F4]], %[[F4]] : f32
  %f5 = subf %f4, %f4 : f32

  // CHECK: %[[I4:.*]] = subi %arg2, %arg2 : i32
  %i4 = "std.subi"(%i, %i) : (i32,i32) -> i32

  // CHECK: %[[I5:.*]] = subi %[[I4]], %[[I4]] : i32
  %i5 = subi %i4, %i4 : i32

  // CHECK: %[[F6:.*]] = mulf %[[F2]], %[[F2]] : f32
  %f6 = mulf %f2, %f2 : f32

  // CHECK: %[[I6:.*]] = muli %[[I2]], %[[I2]] : i32
  %i6 = muli %i2, %i2 : i32

  // CHECK: %c42_i32 = constant 42 : i32
  %x = "std.constant"(){value = 42 : i32} : () -> i32

  // CHECK: %c42_i32_0 = constant 42 : i32
  %7 = constant 42 : i32

  // CHECK: %c43 = constant {crazy = "std.foo"} 43 : index
  %8 = constant {crazy = "std.foo"} 43: index

  // CHECK: %cst = constant 4.300000e+01 : bf16
  %9 = constant 43.0 : bf16

  // CHECK: %f = constant @func_with_ops : (f32) -> ()
  %10 = constant @func_with_ops : (f32) -> ()

  // CHECK: %f_1 = constant @affine_apply : () -> ()
  %11 = constant @affine_apply : () -> ()

  // CHECK: %f_2 = constant @affine_apply : () -> ()
  %12 = constant @affine_apply : () -> ()

  // CHECK: %cst_3 = constant dense<0> : vector<4xi32>
  %13 = constant dense<0> : vector<4 x i32>

  // CHECK: %cst_4 = constant dense<0> : tensor<42xi32>
  %tci32 = constant dense<0> : tensor<42 x i32>

  // CHECK: %cst_5 = constant dense<0> : vector<42xi32>
  %vci32 = constant dense<0> : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = cmpi eq, %{{[0-9]+}}, %{{[0-9]+}} : i32
  %14 = cmpi eq, %i3, %i4 : i32

  // Predicate 1 means inequality comparison.
  // CHECK: %{{[0-9]+}} = cmpi ne, %{{[0-9]+}}, %{{[0-9]+}} : i32
  %15 = "std.cmpi"(%i3, %i4) {predicate = 1} : (i32, i32) -> i1

  // CHECK: %{{[0-9]+}} = cmpi slt, %cst_3, %cst_3 : vector<4xi32>
  %16 = cmpi slt, %13, %13 : vector<4 x i32>

  // CHECK: %{{[0-9]+}} = cmpi ne, %cst_3, %cst_3 : vector<4xi32>
  %17 = "std.cmpi"(%13, %13) {predicate = 1} : (vector<4 x i32>, vector<4 x i32>) -> vector<4 x i1>

  // CHECK: %{{[0-9]+}} = cmpi slt, %arg3, %arg3 : index
  %18 = cmpi slt, %idx, %idx : index

  // CHECK: %{{[0-9]+}} = cmpi eq, %cst_4, %cst_4 : tensor<42xi32>
  %19 = cmpi eq, %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = cmpi eq, %cst_5, %cst_5 : vector<42xi32>
  %20 = cmpi eq, %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %arg3, %arg3 : index
  %21 = select %18, %idx, %idx : index

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %cst_4, %cst_4 : tensor<42xi1>, tensor<42xi32>
  %22 = select %19, %tci32, %tci32 : tensor<42 x i1>, tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %cst_5, %cst_5 : vector<42xi1>, vector<42xi32>
  %23 = select %20, %vci32, %vci32 : vector<42 x i1>, vector<42 x i32>

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %arg3, %arg3 : index
  %24 = "std.select"(%18, %idx, %idx) : (i1, index, index) -> index

  // CHECK: %{{[0-9]+}} = select %{{[0-9]+}}, %cst_4, %cst_4 : tensor<42xi32>
  %25 = std.select %18, %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = divi_signed %arg2, %arg2 : i32
  %26 = divi_signed %i, %i : i32

  // CHECK: %{{[0-9]+}} = divi_signed %arg3, %arg3 : index
  %27 = divi_signed %idx, %idx : index

  // CHECK: %{{[0-9]+}} = divi_signed %cst_5, %cst_5 : vector<42xi32>
  %28 = divi_signed %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = divi_signed %cst_4, %cst_4 : tensor<42xi32>
  %29 = divi_signed %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = divi_signed %arg2, %arg2 : i32
  %30 = "std.divi_signed"(%i, %i) : (i32, i32) -> i32

  // CHECK: %{{[0-9]+}} = divi_unsigned %arg2, %arg2 : i32
  %31 = divi_unsigned %i, %i : i32

  // CHECK: %{{[0-9]+}} = divi_unsigned %arg3, %arg3 : index
  %32 = divi_unsigned %idx, %idx : index

  // CHECK: %{{[0-9]+}} = divi_unsigned %cst_5, %cst_5 : vector<42xi32>
  %33 = divi_unsigned %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = divi_unsigned %cst_4, %cst_4 : tensor<42xi32>
  %34 = divi_unsigned %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = divi_unsigned %arg2, %arg2 : i32
  %35 = "std.divi_unsigned"(%i, %i) : (i32, i32) -> i32

  // CHECK: %{{[0-9]+}} = remi_signed %arg2, %arg2 : i32
  %36 = remi_signed %i, %i : i32

  // CHECK: %{{[0-9]+}} = remi_signed %arg3, %arg3 : index
  %37 = remi_signed %idx, %idx : index

  // CHECK: %{{[0-9]+}} = remi_signed %cst_5, %cst_5 : vector<42xi32>
  %38 = remi_signed %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = remi_signed %cst_4, %cst_4 : tensor<42xi32>
  %39 = remi_signed %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = remi_signed %arg2, %arg2 : i32
  %40 = "std.remi_signed"(%i, %i) : (i32, i32) -> i32

  // CHECK: %{{[0-9]+}} = remi_unsigned %arg2, %arg2 : i32
  %41 = remi_unsigned %i, %i : i32

  // CHECK: %{{[0-9]+}} = remi_unsigned %arg3, %arg3 : index
  %42 = remi_unsigned %idx, %idx : index

  // CHECK: %{{[0-9]+}} = remi_unsigned %cst_5, %cst_5 : vector<42xi32>
  %43 = remi_unsigned %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = remi_unsigned %cst_4, %cst_4 : tensor<42xi32>
  %44 = remi_unsigned %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = remi_unsigned %arg2, %arg2 : i32
  %45 = "std.remi_unsigned"(%i, %i) : (i32, i32) -> i32

  // CHECK: %{{[0-9]+}} = divf %arg1, %arg1 : f32
  %46 = "std.divf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %{{[0-9]+}} = divf %arg1, %arg1 : f32
  %47 = divf %f, %f : f32

  // CHECK: %{{[0-9]+}} = divf %arg0, %arg0 : tensor<4x4x?xf32>
  %48 = divf %t, %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = remf %arg1, %arg1 : f32
  %49 = "std.remf"(%f, %f) : (f32,f32) -> f32

  // CHECK: %{{[0-9]+}} = remf %arg1, %arg1 : f32
  %50 = remf %f, %f : f32

  // CHECK: %{{[0-9]+}} = remf %arg0, %arg0 : tensor<4x4x?xf32>
  %51 = remf %t, %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = and %arg2, %arg2 : i32
  %52 = "std.and"(%i, %i) : (i32,i32) -> i32

  // CHECK: %{{[0-9]+}} = and %arg2, %arg2 : i32
  %53 = and %i, %i : i32

  // CHECK: %{{[0-9]+}} = and %cst_5, %cst_5 : vector<42xi32>
  %54 = std.and %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = and %cst_4, %cst_4 : tensor<42xi32>
  %55 = and %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = or %arg2, %arg2 : i32
  %56 = "std.or"(%i, %i) : (i32,i32) -> i32

  // CHECK: %{{[0-9]+}} = or %arg2, %arg2 : i32
  %57 = or %i, %i : i32

  // CHECK: %{{[0-9]+}} = or %cst_5, %cst_5 : vector<42xi32>
  %58 = std.or %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = or %cst_4, %cst_4 : tensor<42xi32>
  %59 = or %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = xor %arg2, %arg2 : i32
  %60 = "std.xor"(%i, %i) : (i32,i32) -> i32

  // CHECK: %{{[0-9]+}} = xor %arg2, %arg2 : i32
  %61 = xor %i, %i : i32

  // CHECK: %{{[0-9]+}} = xor %cst_5, %cst_5 : vector<42xi32>
  %62 = std.xor %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = xor %cst_4, %cst_4 : tensor<42xi32>
  %63 = xor %tci32, %tci32 : tensor<42 x i32>

  %64 = constant dense<0.> : vector<4 x f32>
  %tcf32 = constant dense<0.> : tensor<42 x f32>
  %vcf32 = constant dense<0.> : vector<4 x f32>

  // CHECK: %{{[0-9]+}} = cmpf ogt, %{{[0-9]+}}, %{{[0-9]+}} : f32
  %65 = cmpf ogt, %f3, %f4 : f32

  // Predicate 0 means ordered equality comparison.
  // CHECK: %{{[0-9]+}} = cmpf oeq, %{{[0-9]+}}, %{{[0-9]+}} : f32
  %66 = "std.cmpf"(%f3, %f4) {predicate = 1} : (f32, f32) -> i1

  // CHECK: %{{[0-9]+}} = cmpf olt, %cst_8, %cst_8 : vector<4xf32>
  %67 = cmpf olt, %vcf32, %vcf32 : vector<4 x f32>

  // CHECK: %{{[0-9]+}} = cmpf oeq, %cst_8, %cst_8 : vector<4xf32>
  %68 = "std.cmpf"(%vcf32, %vcf32) {predicate = 1} : (vector<4 x f32>, vector<4 x f32>) -> vector<4 x i1>

  // CHECK: %{{[0-9]+}} = cmpf oeq, %cst_7, %cst_7 : tensor<42xf32>
  %69 = cmpf oeq, %tcf32, %tcf32 : tensor<42 x f32>

  // CHECK: %{{[0-9]+}} = cmpf oeq, %cst_8, %cst_8 : vector<4xf32>
  %70 = cmpf oeq, %vcf32, %vcf32 : vector<4 x f32>

  // CHECK: %{{[0-9]+}} = rank %arg0 : tensor<4x4x?xf32>
  %71 = "std.rank"(%t) : (tensor<4x4x?xf32>) -> index

  // CHECK: %{{[0-9]+}} = rank %arg0 : tensor<4x4x?xf32>
  %72 = rank %t : tensor<4x4x?xf32>

  // CHECK: = constant unit
  %73 = constant unit

  // CHECK: constant true
  %74 = constant true

  // CHECK: constant false
  %75 = constant false

  // CHECK: = index_cast {{.*}} : index to i64
  %76 = index_cast %idx : index to i64

  // CHECK: = index_cast {{.*}} : i32 to index
  %77 = index_cast %i : i32 to index

  // CHECK: = sitofp {{.*}} : i32 to f32
  %78 = sitofp %i : i32 to f32

  // CHECK: = sitofp {{.*}} : i32 to f64
  %79 = sitofp %i : i32 to f64

  // CHECK: = sitofp {{.*}} : i64 to f32
  %80 = sitofp %j : i64 to f32

  // CHECK: = sitofp {{.*}} : i64 to f64
  %81 = sitofp %j : i64 to f64

  // CHECK: = sexti %arg2 : i32 to i64
  %82 = "std.sexti"(%i) : (i32) -> i64

  // CHECK: = sexti %arg2 : i32 to i64
  %83 = sexti %i : i32 to i64

  // CHECK: %{{[0-9]+}} = sexti %cst_5 : vector<42xi32>
  %84 = sexti %vci32 : vector<42 x i32> to vector<42 x i64>

  // CHECK: %{{[0-9]+}} = sexti %cst_4 : tensor<42xi32>
  %85 = sexti %tci32 : tensor<42 x i32> to tensor<42 x i64>

  // CHECK: = zexti %arg2 : i32 to i64
  %86 = "std.zexti"(%i) : (i32) -> i64

  // CHECK: = zexti %arg2 : i32 to i64
  %87 = zexti %i : i32 to i64

  // CHECK: %{{[0-9]+}} = zexti %cst_5 : vector<42xi32>
  %88 = zexti %vci32 : vector<42 x i32> to vector<42 x i64>

  // CHECK: %{{[0-9]+}} = zexti %cst_4 : tensor<42xi32>
  %89 = zexti %tci32 : tensor<42 x i32> to tensor<42 x i64>

  // CHECK: = trunci %arg2 : i32 to i16
  %90 = "std.trunci"(%i) : (i32) -> i16

  // CHECK: = trunci %arg2 : i32 to i16
  %91 = trunci %i : i32 to i16

  // CHECK: %{{[0-9]+}} = trunci %cst_5 : vector<42xi32>
  %92 = trunci %vci32 : vector<42 x i32> to vector<42 x i16>

  // CHECK: %{{[0-9]+}} = trunci %cst_4 : tensor<42xi32>
  %93 = trunci %tci32 : tensor<42 x i32> to tensor<42 x i16>

  // CHECK: = fpext {{.*}} : f16 to f32
  %94 = fpext %half : f16 to f32

  // CHECK: = fptrunc {{.*}} : f32 to f16
  %95 = fptrunc %f : f32 to f16

  // CHECK: %{{[0-9]+}} = absf %arg1 : f32
  %100 = "std.absf"(%f) : (f32) -> f32

  // CHECK: %{{[0-9]+}} = absf %arg1 : f32
  %101 = absf %f : f32

  // CHECK: %{{[0-9]+}} = absf %cst_8 : vector<4xf32>
  %102 = absf %vcf32 : vector<4xf32>

  // CHECK: %{{[0-9]+}} = absf %arg0 : tensor<4x4x?xf32>
  %103 = absf %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = ceilf %arg1 : f32
  %104 = "std.ceilf"(%f) : (f32) -> f32

  // CHECK: %{{[0-9]+}} = ceilf %arg1 : f32
  %105 = ceilf %f : f32

  // CHECK: %{{[0-9]+}} = ceilf %cst_8 : vector<4xf32>
  %106 = ceilf %vcf32 : vector<4xf32>

  // CHECK: %{{[0-9]+}} = ceilf %arg0 : tensor<4x4x?xf32>
  %107 = ceilf %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = negf %arg1 : f32
  %112 = "std.negf"(%f) : (f32) -> f32

  // CHECK: %{{[0-9]+}} = negf %arg1 : f32
  %113 = negf %f : f32

  // CHECK: %{{[0-9]+}} = negf %cst_8 : vector<4xf32>
  %114 = negf %vcf32 : vector<4xf32>

  // CHECK: %{{[0-9]+}} = negf %arg0 : tensor<4x4x?xf32>
  %115 = negf %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = copysign %arg1, %arg1 : f32
  %116 = "std.copysign"(%f, %f) : (f32, f32) -> f32

  // CHECK: %{{[0-9]+}} = copysign %arg1, %arg1 : f32
  %117 = copysign %f, %f : f32

  // CHECK: %{{[0-9]+}} = copysign %cst_8, %cst_8 : vector<4xf32>
  %118 = copysign %vcf32, %vcf32 : vector<4xf32>

  // CHECK: %{{[0-9]+}} = copysign %arg0, %arg0 : tensor<4x4x?xf32>
  %119 = copysign %t, %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = shift_left %arg2, %arg2 : i32
  %124 = "std.shift_left"(%i, %i) : (i32, i32) -> i32

  // CHECK:%{{[0-9]+}} = shift_left %[[I2]], %[[I2]] : i32
  %125 = shift_left %i2, %i2 : i32

  // CHECK: %{{[0-9]+}} = shift_left %arg3, %arg3 : index
  %126 = shift_left %idx, %idx : index

  // CHECK: %{{[0-9]+}} = shift_left %cst_5, %cst_5 : vector<42xi32>
  %127 = shift_left %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = shift_left %cst_4, %cst_4 : tensor<42xi32>
  %128 = shift_left %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = shift_right_signed %arg2, %arg2 : i32
  %129 = "std.shift_right_signed"(%i, %i) : (i32, i32) -> i32

  // CHECK:%{{[0-9]+}} = shift_right_signed %[[I2]], %[[I2]] : i32
  %130 = shift_right_signed %i2, %i2 : i32

  // CHECK: %{{[0-9]+}} = shift_right_signed %arg3, %arg3 : index
  %131 = shift_right_signed %idx, %idx : index

  // CHECK: %{{[0-9]+}} = shift_right_signed %cst_5, %cst_5 : vector<42xi32>
  %132 = shift_right_signed %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = shift_right_signed %cst_4, %cst_4 : tensor<42xi32>
  %133 = shift_right_signed %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = shift_right_unsigned %arg2, %arg2 : i32
  %134 = "std.shift_right_unsigned"(%i, %i) : (i32, i32) -> i32

  // CHECK:%{{[0-9]+}} = shift_right_unsigned %[[I2]], %[[I2]] : i32
  %135 = shift_right_unsigned %i2, %i2 : i32

  // CHECK: %{{[0-9]+}} = shift_right_unsigned %arg3, %arg3 : index
  %136 = shift_right_unsigned %idx, %idx : index

  // CHECK: %{{[0-9]+}} = shift_right_unsigned %cst_5, %cst_5 : vector<42xi32>
  %137 = shift_right_unsigned %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = shift_right_unsigned %cst_4, %cst_4 : tensor<42xi32>
  %138 = shift_right_unsigned %tci32, %tci32 : tensor<42 x i32>

  // CHECK: = fpext {{.*}} : vector<4xf32> to vector<4xf64>
  %143 = fpext %vcf32 : vector<4xf32> to vector<4xf64>

  // CHECK: = fptrunc {{.*}} : vector<4xf32> to vector<4xf16>
  %144 = fptrunc %vcf32 : vector<4xf32> to vector<4xf16>

  // CHECK: %{{[0-9]+}} = math.rsqrt %arg1 : f32
  %145 = math.rsqrt %f : f32

  // CHECK: = fptosi {{.*}} : f32 to i32
  %159 = fptosi %f : f32 to i32

  // CHECK: = fptosi {{.*}} : f32 to i64
  %160 = fptosi %f : f32 to i64

  // CHECK: = fptosi {{.*}} : f16 to i32
  %161 = fptosi %half : f16 to i32

  // CHECK: = fptosi {{.*}} : f16 to i64
  %162 = fptosi %half : f16 to i64

  // CHECK: floorf %arg1 : f32
  %163 = "std.floorf"(%f) : (f32) -> f32

  // CHECK: %{{[0-9]+}} = floorf %arg1 : f32
  %164 = floorf %f : f32

  // CHECK: %{{[0-9]+}} = floorf %cst_8 : vector<4xf32>
  %165 = floorf %vcf32 : vector<4xf32>

  // CHECK: %{{[0-9]+}} = floorf %arg0 : tensor<4x4x?xf32>
  %166 = floorf %t : tensor<4x4x?xf32>

  // CHECK: %{{[0-9]+}} = floordivi_signed %arg2, %arg2 : i32
  %167 = floordivi_signed %i, %i : i32

  // CHECK: %{{[0-9]+}} = floordivi_signed %arg3, %arg3 : index
  %168 = floordivi_signed %idx, %idx : index

  // CHECK: %{{[0-9]+}} = floordivi_signed %cst_5, %cst_5 : vector<42xi32>
  %169 = floordivi_signed %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = floordivi_signed %cst_4, %cst_4 : tensor<42xi32>
  %170 = floordivi_signed %tci32, %tci32 : tensor<42 x i32>

  // CHECK: %{{[0-9]+}} = ceildivi_signed %arg2, %arg2 : i32
  %171 = ceildivi_signed %i, %i : i32

  // CHECK: %{{[0-9]+}} = ceildivi_signed %arg3, %arg3 : index
  %172 = ceildivi_signed %idx, %idx : index

  // CHECK: %{{[0-9]+}} = ceildivi_signed %cst_5, %cst_5 : vector<42xi32>
  %173 = ceildivi_signed %vci32, %vci32 : vector<42 x i32>

  // CHECK: %{{[0-9]+}} = ceildivi_signed %cst_4, %cst_4 : tensor<42xi32>
  %174 = ceildivi_signed %tci32, %tci32 : tensor<42 x i32>

  return
}

// CHECK-LABEL: func @affine_apply() {
func @affine_apply() {
  %i = "std.constant"() {value = 0: index} : () -> index
  %j = "std.constant"() {value = 1: index} : () -> index

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
  // CHECK: %0 = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %2 = "std.load"(%0, %1, %1) : (memref<4x4xi32>, index, index)->i32

  // CHECK: %{{.*}} = load %arg0[%arg1, %arg1] : memref<4x4xi32>
  %3 = load %0[%1, %1] : memref<4x4xi32>

  // CHECK: prefetch %arg0[%arg1, %arg1], write, locality<1>, data : memref<4x4xi32>
  prefetch %0[%1, %1], write, locality<1>, data : memref<4x4xi32>

  // CHECK: prefetch %arg0[%arg1, %arg1], read, locality<3>, instr : memref<4x4xi32>
  prefetch %0[%1, %1], read, locality<3>, instr : memref<4x4xi32>

  return
}

// Test with zero-dimensional operands using no index in load/store.
// CHECK-LABEL: func @zero_dim_no_idx
func @zero_dim_no_idx(%arg0 : memref<i32>, %arg1 : memref<i32>, %arg2 : memref<i32>) {
  %0 = std.load %arg0[] : memref<i32>
  std.store %0, %arg1[] : memref<i32>
  return
  // CHECK: %0 = load %{{.*}}[] : memref<i32>
  // CHECK: store %{{.*}}, %{{.*}}[] : memref<i32>
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
  // CHECK: %0 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>
  %0 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>

  // CHECK: %1 = memref_cast %arg1 : memref<?xf32> to memref<4xf32>
  %1 = memref_cast %arg1 : memref<?xf32> to memref<4xf32>

  // CHECK: {{%.*}} = memref_cast %arg2 : memref<64x16x4xf32, #[[$BASE_MAP0]]> to memref<64x16x4xf32, #[[$BASE_MAP3]]>
  %2 = memref_cast %arg2 : memref<64x16x4xf32, offset: 0, strides: [64, 4, 1]> to memref<64x16x4xf32, offset: ?, strides: [?, ?, ?]>

  // CHECK: {{%.*}} = memref_cast {{%.*}} : memref<64x16x4xf32, #[[$BASE_MAP3]]> to memref<64x16x4xf32, #[[$BASE_MAP0]]>
  %3 = memref_cast %2 : memref<64x16x4xf32, offset: ?, strides: [?, ?, ?]> to memref<64x16x4xf32, offset: 0, strides: [64, 4, 1]>

  // CHECK: memref_cast %{{.*}} : memref<4xf32> to memref<*xf32>
  %4 = memref_cast %1 : memref<4xf32> to memref<*xf32>

  // CHECK: memref_cast %{{.*}} : memref<*xf32> to memref<4xf32>
  %5 = memref_cast %4 : memref<*xf32> to memref<4xf32>
  return
}

// Check that unranked memrefs with non-default memory space roundtrip
// properly.
// CHECK-LABEL: @unranked_memref_roundtrip(memref<*xf32, 4>)
func private @unranked_memref_roundtrip(memref<*xf32, 4>)

// CHECK-LABEL: func @memref_view(%arg0
func @memref_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8>
  // Test two dynamic sizes and dynamic offset.
  // CHECK: %{{.*}} = std.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>
  %1 = view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>

  // Test one dynamic size and dynamic offset.
  // CHECK: %{{.*}} = std.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>
  %3 = view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>

  // Test static sizes and static offset.
  // CHECK: %{{.*}} = std.view %0[{{.*}}][] : memref<2048xi8> to memref<64x4xf32>
  %c0 = constant 0: index
  %5 = view %0[%c0][] : memref<2048xi8> to memref<64x4xf32>
  return
}

// CHECK-LABEL: func @memref_subview(%arg0
func @memref_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %0 = alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>
  // CHECK: subview %0[%c0, %c0, %c0] [%arg0, %arg1, %arg2] [%c1, %c1, %c1] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]>
  // CHECK-SAME: to memref<?x?x?xf32, #[[$BASE_MAP3]]>
  %1 = subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1]
    : memref<8x16x4xf32, offset:0, strides: [64, 4, 1]> to
      memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>

  %2 = alloc()[%arg2] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  // CHECK: subview %2[%c1] [%arg0] [%c1] :
  // CHECK-SAME: memref<64xf32, #[[$BASE_MAP1]]>
  // CHECK-SAME: to memref<?xf32, #[[$SUBVIEW_MAP1]]>
  %3 = subview %2[%c1][%arg0][%c1]
    : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to
      memref<?xf32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>

  %4 = alloc() : memref<64x22xf32, affine_map<(d0, d1) -> (d0 * 22 + d1)>>
  // CHECK: subview %4[%c0, %c1] [%arg0, %arg1] [%c1, %c0] :
  // CHECK-SAME: memref<64x22xf32, #[[$BASE_MAP2]]>
  // CHECK-SAME: to memref<?x?xf32, #[[$SUBVIEW_MAP2]]>
  %5 = subview %4[%c0, %c1][%arg0, %arg1][%c1, %c0]
    : memref<64x22xf32, offset:0, strides: [22, 1]> to
      memref<?x?xf32, offset:?, strides: [?, ?]>

  // CHECK: subview %0[0, 2, 0] [4, 4, 4] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, #[[$BASE_MAP0]]>
  // CHECK-SAME: to memref<4x4x4xf32, #[[$SUBVIEW_MAP3]]>
  %6 = subview %0[0, 2, 0][4, 4, 4][1, 1, 1]
    : memref<8x16x4xf32, offset:0, strides: [64, 4, 1]> to
      memref<4x4x4xf32, offset:8, strides: [64, 4, 1]>

  %7 = alloc(%arg1, %arg2) : memref<?x?xf32>
  // CHECK: subview {{%.*}}[0, 0] [4, 4] [1, 1] :
  // CHECK-SAME: memref<?x?xf32>
  // CHECK-SAME: to memref<4x4xf32, #[[$SUBVIEW_MAP4]]>
  %8 = subview %7[0, 0][4, 4][1, 1]
    : memref<?x?xf32> to memref<4x4xf32, offset: ?, strides:[?, 1]>

  %9 = alloc() : memref<16x4xf32>
  // CHECK: subview {{%.*}}[{{%.*}}, {{%.*}}] [4, 4] [{{%.*}}, {{%.*}}] :
  // CHECK-SAME: memref<16x4xf32>
  // CHECK-SAME: to memref<4x4xf32, #[[$SUBVIEW_MAP2]]
  %10 = subview %9[%arg1, %arg1][4, 4][%arg2, %arg2]
    : memref<16x4xf32> to memref<4x4xf32, offset: ?, strides:[?, ?]>

  // CHECK: subview {{%.*}}[{{%.*}}, {{%.*}}] [4, 4] [2, 2] :
  // CHECK-SAME: memref<16x4xf32>
  // CHECK-SAME: to memref<4x4xf32, #[[$SUBVIEW_MAP5]]
  %11 = subview %9[%arg1, %arg2][4, 4][2, 2]
    : memref<16x4xf32> to memref<4x4xf32, offset: ?, strides:[8, 2]>

  %12 = alloc() : memref<1x9x1x4x1xf32, affine_map<(d0, d1, d2, d3, d4) -> (36 * d0 + 36 * d1 + 4 * d2 + 4 * d3 + d4)>>
  // CHECK: subview %12[%arg1, %arg1, %arg1, %arg1, %arg1]
  // CHECK-SAME: [1, 9, 1, 4, 1] [%arg2, %arg2, %arg2, %arg2, %arg2] :
  // CHECK-SAME: memref<1x9x1x4x1xf32, #[[$SUBVIEW_MAP6]]> to memref<9x4xf32, #[[$SUBVIEW_MAP2]]>
  %13 = subview %12[%arg1, %arg1, %arg1, %arg1, %arg1][1, 9, 1, 4, 1][%arg2, %arg2, %arg2, %arg2, %arg2] : memref<1x9x1x4x1xf32, offset: 0, strides: [36, 36, 4, 4, 1]> to memref<9x4xf32, offset: ?, strides: [?, ?]>
  // CHECK: subview %12[%arg1, %arg1, %arg1, %arg1, %arg1]
  // CHECK-SAME: [1, 9, 1, 4, 1] [%arg2, %arg2, %arg2, %arg2, %arg2] :
  // CHECK-SAME: memref<1x9x1x4x1xf32, #[[$SUBVIEW_MAP6]]> to memref<1x9x4xf32, #[[$BASE_MAP3]]>
  %14 = subview %12[%arg1, %arg1, %arg1, %arg1, %arg1][1, 9, 1, 4, 1][%arg2, %arg2, %arg2, %arg2, %arg2] : memref<1x9x1x4x1xf32, offset: 0, strides: [36, 36, 4, 4, 1]> to memref<1x9x4xf32, offset: ?, strides: [?, ?, ?]>

  %15 = alloc(%arg1, %arg2)[%c0, %c1, %arg1, %arg0, %arg0, %arg2, %arg2] : memref<1x?x5x1x?x1xf32, affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6] -> (s0 + s1 * d0 + s2 * d1 + s3 * d2 + s4 * d3 + s5 * d4 + s6 * d5)>>
  // CHECK: subview %15[0, 0, 0, 0, 0, 0] [1, %arg1, 5, 1, %arg2, 1] [1, 1, 1, 1, 1, 1]  :
  // CHECK-SAME: memref<1x?x5x1x?x1xf32,  #[[$SUBVIEW_MAP7]]> to memref<?x5x?xf32, #[[$BASE_MAP3]]>
  %16 = subview %15[0, 0, 0, 0, 0, 0][1, %arg1, 5, 1, %arg2, 1][1, 1, 1, 1, 1, 1] : memref<1x?x5x1x?x1xf32, offset: ?, strides: [?, ?, ?, ?, ?, ?]> to memref<?x5x?xf32, offset: ?, strides: [?, ?, ?]>
  // CHECK: subview %15[%arg1, %arg1, %arg1, %arg1, %arg1, %arg1] [1, %arg1, 5, 1, %arg2, 1] [1, 1, 1, 1, 1, 1]  :
  // CHECK-SAME: memref<1x?x5x1x?x1xf32, #[[$SUBVIEW_MAP7]]> to memref<?x5x?x1xf32, #[[$SUBVIEW_MAP8]]>
  %17 = subview %15[%arg1, %arg1, %arg1, %arg1, %arg1, %arg1][1, %arg1, 5, 1, %arg2, 1][1, 1, 1, 1, 1, 1] :  memref<1x?x5x1x?x1xf32, offset: ?, strides: [?, ?, ?, ?, ?, ?]> to memref<?x5x?x1xf32, offset: ?, strides: [?, ?, ?, ?]>

  %18 = alloc() : memref<1x8xf32>
  // CHECK: subview %18[0, 0] [1, 8] [1, 1]  : memref<1x8xf32> to memref<8xf32>
  %19 = subview %18[0, 0][1, 8][1, 1] : memref<1x8xf32> to memref<8xf32>

  %20 = alloc() : memref<8x16x4xf32>
  // CHECK: subview %20[0, 0, 0] [1, 16, 4] [1, 1, 1]  : memref<8x16x4xf32> to memref<16x4xf32>
  %21 = subview %20[0, 0, 0][1, 16, 4][1, 1, 1] : memref<8x16x4xf32> to memref<16x4xf32>

  %22 = subview %20[3, 4, 2][1, 6, 3][1, 1, 1] : memref<8x16x4xf32> to memref<6x3xf32, offset: 210, strides: [4, 1]>

  %23 = alloc() : memref<f32>
  %78 = subview %23[] [] []  : memref<f32> to memref<f32>

  /// Subview with only leading operands.
  %24 = alloc() : memref<5x3xf32>
  // CHECK: subview %{{.*}}[2] [3] [1] : memref<5x3xf32> to memref<3x3xf32, #[[$SUBVIEW_MAP9]]>
  %25 = subview %24[2][3][1]: memref<5x3xf32> to memref<3x3xf32, offset: 6, strides: [3, 1]>

  /// Rank-reducing subview with only leading operands.
  // CHECK: subview %{{.*}}[1] [1] [1] : memref<5x3xf32> to memref<3xf32, #[[$SUBVIEW_MAP10]]>
  %26 = subview %24[1][1][1]: memref<5x3xf32> to memref<3xf32, offset: 3, strides: [1]>

  // Corner-case of 0-D rank-reducing subview with an offset.
  // CHECK: subview %{{.*}}[1, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, #[[$SUBVIEW_MAP11]]>
  %27 = subview %24[1, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, affine_map<() -> (4)>>

  // CHECK: subview %{{.*}}[%{{.*}}, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, #[[$SUBVIEW_MAP12]]>
  %28 = subview %24[%arg0, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, affine_map<()[s0] -> (s0)>>

  // CHECK: subview %{{.*}}[0, %{{.*}}] [%{{.*}}, 1] [1, 1] : memref<?x?xf32> to memref<?xf32, #[[$SUBVIEW_MAP1]]>
  %a30 = alloc(%arg0, %arg0) : memref<?x?xf32>
  %30 = subview %a30[0, %arg1][%arg2, 1][1, 1] : memref<?x?xf32> to memref<?xf32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>

  return
}

// CHECK-LABEL: func @test_dimop
// CHECK-SAME: %[[ARG:.*]]: tensor<4x4x?xf32>
func @test_dimop(%arg0: tensor<4x4x?xf32>) {
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: %{{.*}} = dim %[[ARG]], %[[C2]] : tensor<4x4x?xf32>
  %c2 = constant 2 : index
  %0 = dim %arg0, %c2 : tensor<4x4x?xf32>
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
  // CHECK: %[[TENSOR:.*]] = tensor_load %[[MEMREF:.*]] : memref<4x4xi32>
  %1 = tensor_load %0 : memref<4x4xi32>
  // CHECK: tensor_store %[[TENSOR]], %[[MEMREF]] : memref<4x4xi32>
  tensor_store %1, %0 : memref<4x4xi32>
  return
}

// CHECK-LABEL: func @unranked_tensor_load_store
func @unranked_tensor_load_store(%0 : memref<*xi32>) {
  // CHECK: %[[TENSOR:.*]] = tensor_load %[[MEMREF:.*]] : memref<*xi32>
  %1 = tensor_load %0 : memref<*xi32>
  // CHECK: tensor_store %[[TENSOR]], %[[MEMREF]] : memref<*xi32>
  tensor_store %1, %0 : memref<*xi32>
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
      %c1 = constant 1.0 : f32
      %out = addf %c1, %old_value : f32
      atomic_yield %out : f32
  // CHECK: index_attr = 8 : index
  } { index_attr = 8 : index }
  return
}

// CHECK-LABEL: func @assume_alignment
// CHECK-SAME: %[[MEMREF:.*]]: memref<4x4xf16>
func @assume_alignment(%0: memref<4x4xf16>) {
  // CHECK: assume_alignment %[[MEMREF]], 16 : memref<4x4xf16>
  assume_alignment %0, 16 : memref<4x4xf16>
  return
}

// CHECK-LABEL: func @subtensor({{.*}}) {
func @subtensor(%t: tensor<8x16x4xf32>, %idx : index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // CHECK: subtensor
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<?x?x?xf32>
  %1 = subtensor %t[%c0, %c0, %c0][%idx, %idx, %idx][%c1, %c1, %c1]
    : tensor<8x16x4xf32> to tensor<?x?x?xf32>

  // CHECK: subtensor
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<4x4x4xf32>
  %2 = subtensor %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>

  // CHECK: subtensor
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<4x4xf32>
  %3 = subtensor %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4xf32>

  return
}

// CHECK-LABEL: func @subtensor_insert({{.*}}) {
func @subtensor_insert(
    %t: tensor<8x16x4xf32>,
    %t2: tensor<16x32x8xf32>,
    %t3: tensor<4x4xf32>,
    %idx : index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // CHECK: subtensor_insert
  // CHECK-SAME: tensor<8x16x4xf32> into tensor<16x32x8xf32>
  %1 = subtensor_insert %t into %t2[%c0, %c0, %c0][%idx, %idx, %idx][%c1, %c1, %c1]
    : tensor<8x16x4xf32> into tensor<16x32x8xf32>

  // CHECK: subtensor_insert
  // CHECK-SAME: tensor<8x16x4xf32> into tensor<16x32x8xf32>
  %2 = subtensor_insert %t into %t2[%c0, %idx, %c0][%idx, 4, %idx][%c1, 1, %c1]
    : tensor<8x16x4xf32> into tensor<16x32x8xf32>

  // CHECK: subtensor_insert
  // CHECK-SAME: tensor<4x4xf32> into tensor<8x16x4xf32>
  %3 = subtensor_insert %t3 into %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : tensor<4x4xf32> into tensor<8x16x4xf32>

  return
}
