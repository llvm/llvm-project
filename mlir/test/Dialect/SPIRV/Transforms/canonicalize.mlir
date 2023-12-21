// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.AccessChain
//===----------------------------------------------------------------------===//

func.func @combine_full_access_chain() -> f32 {
  // CHECK: %[[INDEX:.*]] = spirv.Constant 0
  // CHECK-NEXT: %[[VAR:.*]] = spirv.Variable
  // CHECK-NEXT: %[[PTR:.*]] = spirv.AccessChain %[[VAR]][%[[INDEX]], %[[INDEX]], %[[INDEX]]]
  // CHECK-NEXT: spirv.Load "Function" %[[PTR]]
  %c0 = spirv.Constant 0: i32
  %0 = spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<4x!spirv.array<4xf32>>, !spirv.array<4xi32>)>, Function>
  %1 = spirv.AccessChain %0[%c0] : !spirv.ptr<!spirv.struct<(!spirv.array<4x!spirv.array<4xf32>>, !spirv.array<4xi32>)>, Function>, i32
  %2 = spirv.AccessChain %1[%c0, %c0] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32, i32
  %3 = spirv.Load "Function" %2 : f32
  spirv.ReturnValue %3 : f32
}

// -----

func.func @combine_access_chain_multi_use() -> !spirv.array<4xf32> {
  // CHECK: %[[INDEX:.*]] = spirv.Constant 0
  // CHECK-NEXT: %[[VAR:.*]] = spirv.Variable
  // CHECK-NEXT: %[[PTR_0:.*]] = spirv.AccessChain %[[VAR]][%[[INDEX]], %[[INDEX]]]
  // CHECK-NEXT: %[[PTR_1:.*]] = spirv.AccessChain %[[VAR]][%[[INDEX]], %[[INDEX]], %[[INDEX]]]
  // CHECK-NEXT: spirv.Load "Function" %[[PTR_0]]
  // CHECK-NEXT: spirv.Load "Function" %[[PTR_1]]
  %c0 = spirv.Constant 0: i32
  %0 = spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<4x!spirv.array<4xf32>>, !spirv.array<4xi32>)>, Function>
  %1 = spirv.AccessChain %0[%c0] : !spirv.ptr<!spirv.struct<(!spirv.array<4x!spirv.array<4xf32>>, !spirv.array<4xi32>)>, Function>, i32
  %2 = spirv.AccessChain %1[%c0] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>, i32
  %3 = spirv.AccessChain %2[%c0] : !spirv.ptr<!spirv.array<4xf32>, Function>, i32
  %4 = spirv.Load "Function" %2 : !spirv.array<4xf32>
  %5 = spirv.Load "Function" %3 : f32
  spirv.ReturnValue %4: !spirv.array<4xf32>
}

// -----

func.func @dont_combine_access_chain_without_common_base() -> !spirv.array<4xi32> {
  // CHECK: %[[INDEX:.*]] = spirv.Constant 1
  // CHECK-NEXT: %[[VAR_0:.*]] = spirv.Variable
  // CHECK-NEXT: %[[VAR_1:.*]] = spirv.Variable
  // CHECK-NEXT: %[[VAR_0_PTR:.*]] = spirv.AccessChain %[[VAR_0]][%[[INDEX]]]
  // CHECK-NEXT: %[[VAR_1_PTR:.*]] = spirv.AccessChain %[[VAR_1]][%[[INDEX]]]
  // CHECK-NEXT: spirv.Load "Function" %[[VAR_0_PTR]]
  // CHECK-NEXT: spirv.Load "Function" %[[VAR_1_PTR]]
  %c1 = spirv.Constant 1: i32
  %0 = spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<4x!spirv.array<4xf32>>, !spirv.array<4xi32>)>, Function>
  %1 = spirv.Variable : !spirv.ptr<!spirv.struct<(!spirv.array<4x!spirv.array<4xf32>>, !spirv.array<4xi32>)>, Function>
  %2 = spirv.AccessChain %0[%c1] : !spirv.ptr<!spirv.struct<(!spirv.array<4x!spirv.array<4xf32>>, !spirv.array<4xi32>)>, Function>, i32
  %3 = spirv.AccessChain %1[%c1] : !spirv.ptr<!spirv.struct<(!spirv.array<4x!spirv.array<4xf32>>, !spirv.array<4xi32>)>, Function>, i32
  %4 = spirv.Load "Function" %2 : !spirv.array<4xi32>
  %5 = spirv.Load "Function" %3 : !spirv.array<4xi32>
  spirv.ReturnValue %4 : !spirv.array<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Bitcast
//===----------------------------------------------------------------------===//

func.func @convert_bitcast_full(%arg0 : vector<2xf32>) -> f64 {
  // CHECK: %[[RESULT:.*]] = spirv.Bitcast {{%.*}} : vector<2xf32> to f64
  // CHECK-NEXT: spirv.ReturnValue %[[RESULT]]
  %0 = spirv.Bitcast %arg0 : vector<2xf32> to vector<2xi32>
  %1 = spirv.Bitcast %0 : vector<2xi32> to i64
  %2 = spirv.Bitcast %1 : i64 to f64
  spirv.ReturnValue %2 : f64
}

// -----

func.func @convert_bitcast_multi_use(%arg0 : vector<2xf32>, %arg1 : !spirv.ptr<i64, Uniform>) -> f64 {
  // CHECK: %[[RESULT_0:.*]] = spirv.Bitcast {{%.*}} : vector<2xf32> to i64
  // CHECK-NEXT: %[[RESULT_1:.*]] = spirv.Bitcast {{%.*}} : vector<2xf32> to f64
  // CHECK-NEXT: spirv.Store {{".*"}} {{%.*}}, %[[RESULT_0]]
  // CHECK-NEXT: spirv.ReturnValue %[[RESULT_1]]
  %0 = spirv.Bitcast %arg0 : vector<2xf32> to i64
  %1 = spirv.Bitcast %0 : i64 to f64
  spirv.Store "Uniform" %arg1, %0 : i64
  spirv.ReturnValue %1 : f64
}

// -----

// CHECK-LABEL: @convert_bitcast_roundtip
// CHECK-SAME:    %[[ARG:.+]]: i64
func.func @convert_bitcast_roundtip(%arg0 : i64) -> i64 {
  // CHECK: spirv.ReturnValue %[[ARG]]
  %0 = spirv.Bitcast %arg0 : i64 to f64
  %1 = spirv.Bitcast %0 : f64 to i64
  spirv.ReturnValue %1 : i64
}

// -----

// CHECK-LABEL: @convert_bitcast_chained_roundtip
// CHECK-SAME:    %[[ARG:.+]]: i64
func.func @convert_bitcast_chained_roundtip(%arg0 : i64) -> i64 {
  // CHECK: spirv.ReturnValue %[[ARG]]
  %0 = spirv.Bitcast %arg0 : i64 to f64
  %1 = spirv.Bitcast %0 : f64 to vector<2xi32>
  %2 = spirv.Bitcast %1 : vector<2xi32> to vector<2xf32>
  %3 = spirv.Bitcast %2 : vector<2xf32> to i64
  spirv.ReturnValue %3 : i64
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CompositeExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: extract_vector
func.func @extract_vector() -> (i32, i32, i32) {
  // CHECK-DAG: spirv.Constant 6 : i32
  // CHECK-DAG: spirv.Constant -33 : i32
  // CHECK-DAG: spirv.Constant 42 : i32
  %0 = spirv.Constant dense<[42, -33, 6]> : vector<3xi32>
  %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
  %2 = spirv.CompositeExtract %0[1 : i32] : vector<3xi32>
  %3 = spirv.CompositeExtract %0[2 : i32] : vector<3xi32>
  return %1, %2, %3 : i32, i32, i32
}

// -----

// CHECK-LABEL: extract_array_final
func.func @extract_array_final() -> (i32, i32) {
  // CHECK-DAG: spirv.Constant -5 : i32
  // CHECK-DAG: spirv.Constant 4 : i32
  %0 = spirv.Constant [dense<[4, -5]> : vector<2xi32>] : !spirv.array<1 x vector<2xi32>>
  %1 = spirv.CompositeExtract %0[0 : i32, 0 : i32] : !spirv.array<1 x vector<2 x i32>>
  %2 = spirv.CompositeExtract %0[0 : i32, 1 : i32] : !spirv.array<1 x vector<2 x i32>>
  return %1, %2 : i32, i32
}

// -----

// CHECK-LABEL: extract_array_interm
func.func @extract_array_interm() -> (vector<2xi32>) {
  // CHECK: spirv.Constant dense<[4, -5]> : vector<2xi32>
  %0 = spirv.Constant [dense<[4, -5]> : vector<2xi32>] : !spirv.array<1 x vector<2xi32>>
  %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.array<1 x vector<2 x i32>>
  return %1 : vector<2xi32>
}

// -----

// CHECK-LABEL: extract_from_not_constant
func.func @extract_from_not_constant() -> i32 {
  %0 = spirv.Variable : !spirv.ptr<vector<3xi32>, Function>
  %1 = spirv.Load "Function" %0 : vector<3xi32>
  // CHECK: spirv.CompositeExtract
  %2 = spirv.CompositeExtract %1[0 : i32] : vector<3xi32>
  spirv.ReturnValue %2 : i32
}

// -----

// CHECK-LABEL: extract_insert
//  CHECK-SAME: (%[[COMP:.+]]: !spirv.array<1 x vector<2xf32>>, %[[VAL:.+]]: f32)
func.func @extract_insert(%composite: !spirv.array<1xvector<2xf32>>, %val: f32) -> (f32, f32) {
  // CHECK: %[[INSERT:.+]] = spirv.CompositeInsert %[[VAL]], %[[COMP]]
  %insert = spirv.CompositeInsert %val, %composite[0 : i32, 1 : i32] : f32 into !spirv.array<1xvector<2xf32>>
  %1 = spirv.CompositeExtract %insert[0 : i32, 0 : i32] : !spirv.array<1xvector<2xf32>>
  // CHECK: %[[S:.+]] = spirv.CompositeExtract %[[INSERT]][0 : i32, 0 : i32]
  %2 = spirv.CompositeExtract %insert[0 : i32, 1 : i32] : !spirv.array<1xvector<2xf32>>
  // CHECK: return %[[S]], %[[VAL]]
  return %1, %2 : f32, f32
}

// -----

// CHECK-LABEL: extract_construct
//  CHECK-SAME: (%[[VAL1:.+]]: vector<2xf32>, %[[VAL2:.+]]: vector<2xf32>)
func.func @extract_construct(%val1: vector<2xf32>, %val2: vector<2xf32>) -> (vector<2xf32>, vector<2xf32>) {
  %construct = spirv.CompositeConstruct %val1, %val2 : (vector<2xf32>, vector<2xf32>) -> !spirv.array<2xvector<2xf32>>
  %1 = spirv.CompositeExtract %construct[0 : i32] : !spirv.array<2xvector<2xf32>>
  %2 = spirv.CompositeExtract %construct[1 : i32] : !spirv.array<2xvector<2xf32>>
  // CHECK: return %[[VAL1]], %[[VAL2]]
  return %1, %2 : vector<2xf32>, vector<2xf32>
}

// -----

 // CHECK-LABEL: fold_composite_op
 //  CHECK-SAME: (%[[COMP:.+]]: !spirv.struct<(f32, f32)>, %[[VAL1:.+]]: f32, %[[VAL2:.+]]: f32)
  func.func @fold_composite_op(%composite: !spirv.struct<(f32, f32)>, %val1: f32, %val2: f32) -> f32 {
    %insert = spirv.CompositeInsert %val1, %composite[0 : i32] : f32 into !spirv.struct<(f32, f32)>
    %1 = spirv.CompositeInsert %val2, %insert[1 : i32] : f32 into !spirv.struct<(f32, f32)>
    %2 = spirv.CompositeExtract %1[0 : i32] : !spirv.struct<(f32, f32)>
    // CHECK-NEXT: return  %[[VAL1]]
    return %2 : f32
  }

// -----

 // CHECK-LABEL: fold_composite_op
 //  CHECK-SAME: (%[[VAL1:.+]]: f32, %[[VAL2:.+]]: f32, %[[VAL3:.+]]: f32)
  func.func @fold_composite_op(%val1: f32, %val2: f32, %val3: f32) -> f32 {
    %composite = spirv.CompositeConstruct %val1, %val1, %val1 : (f32, f32, f32) -> !spirv.struct<(f32, f32, f32)>
    %insert = spirv.CompositeInsert %val2, %composite[1 : i32] : f32 into !spirv.struct<(f32, f32, f32)>
    %1 = spirv.CompositeInsert %val3, %insert[2 : i32] : f32 into !spirv.struct<(f32, f32, f32)>
    %2 = spirv.CompositeExtract %1[0 : i32] : !spirv.struct<(f32, f32, f32)>
    // CHECK-NEXT: return  %[[VAL1]]
    return %2 : f32
  }

// -----

// Not yet implemented case

// CHECK-LABEL: extract_construct
func.func @extract_construct(%val1: vector<3xf32>, %val2: f32) -> (f32, f32) {
  // CHECK: spirv.CompositeConstruct
  %construct = spirv.CompositeConstruct %val1, %val2 : (vector<3xf32>, f32) -> vector<4xf32>
  // CHECK: spirv.CompositeExtract
  %1 = spirv.CompositeExtract %construct[0 : i32] : vector<4xf32>
  // CHECK: spirv.CompositeExtract
  %2 = spirv.CompositeExtract %construct[1 : i32] : vector<4xf32>
  return %1, %2 : f32, f32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Constant
//===----------------------------------------------------------------------===//

// TODO: test constants in different blocks

func.func @deduplicate_scalar_constant() -> (i32, i32) {
  // CHECK: %[[CST:.*]] = spirv.Constant 42 : i32
  %0 = spirv.Constant 42 : i32
  %1 = spirv.Constant 42 : i32
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : i32, i32
}

// -----

func.func @deduplicate_vector_constant() -> (vector<3xi32>, vector<3xi32>) {
  // CHECK: %[[CST:.*]] = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  %0 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : vector<3xi32>, vector<3xi32>
}

// -----

func.func @deduplicate_composite_constant() -> (!spirv.array<1 x vector<2xi32>>, !spirv.array<1 x vector<2xi32>>) {
  // CHECK: %[[CST:.*]] = spirv.Constant [dense<5> : vector<2xi32>] : !spirv.array<1 x vector<2xi32>>
  %0 = spirv.Constant [dense<5> : vector<2xi32>] : !spirv.array<1 x vector<2xi32>>
  %1 = spirv.Constant [dense<5> : vector<2xi32>] : !spirv.array<1 x vector<2xi32>>
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : !spirv.array<1 x vector<2xi32>>, !spirv.array<1 x vector<2xi32>>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iadd_zero
// CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @iadd_zero(%arg0: i32) -> (i32, i32) {
  %zero = spirv.Constant 0 : i32
  %0 = spirv.IAdd %arg0, %zero : i32
  %1 = spirv.IAdd %zero, %arg0 : i32
  // CHECK: return %[[ARG]], %[[ARG]]
  return %0, %1: i32, i32
}

// CHECK-LABEL: @const_fold_scalar_iadd_normal
func.func @const_fold_scalar_iadd_normal() -> (i32, i32, i32) {
  %c5 = spirv.Constant 5 : i32
  %cn8 = spirv.Constant -8 : i32

  // CHECK-DAG: spirv.Constant -3
  // CHECK-DAG: spirv.Constant -16
  // CHECK-DAG: spirv.Constant 10
  %0 = spirv.IAdd %c5, %c5 : i32
  %1 = spirv.IAdd %cn8, %cn8 : i32
  %2 = spirv.IAdd %c5, %cn8 : i32
  return %0, %1, %2: i32, i32, i32
}

// CHECK-LABEL: @const_fold_scalar_iadd_flow
func.func @const_fold_scalar_iadd_flow() -> (i32, i32, i32, i32) {
  %c1 = spirv.Constant 1 : i32
  %c2 = spirv.Constant 2 : i32
  %c3 = spirv.Constant 4294967295 : i32  // 2^32 - 1: 0xffff ffff
  %c4 = spirv.Constant -2147483648 : i32 // -2^31   : 0x8000 0000
  %c5 = spirv.Constant -1 : i32          //         : 0xffff ffff
  %c6 = spirv.Constant -2 : i32          //         : 0xffff fffe

  // 0x8000 0000 + 0xffff fffe = 0x1 7fff fffe -> 0x7fff fffe
  // CHECK-DAG: spirv.Constant 2147483646
  // 0x8000 0000 + 0xffff ffff = 0x1 7fff ffff -> 0x7fff ffff
  // CHECK-DAG: spirv.Constant 2147483647
  // 0x0000 0002 + 0xffff ffff = 0x1 0000 0001 -> 0x0000 0001
  // CHECK-DAG: spirv.Constant 1
  // 0x0000 0001 + 0xffff ffff = 0x1 0000 0000 -> 0x0000 0000
  // CHECK-DAG: spirv.Constant 0
  %0 = spirv.IAdd %c1, %c3 : i32
   %1 = spirv.IAdd %c2, %c3 : i32
  %2 = spirv.IAdd %c4, %c5 : i32
  %3 = spirv.IAdd %c4, %c6 : i32
  return %0, %1, %2, %3: i32, i32, i32, i32
}

// CHECK-LABEL: @const_fold_vector_iadd
func.func @const_fold_vector_iadd() -> vector<3xi32> {
  %vc1 = spirv.Constant dense<[42, -55, 127]> : vector<3xi32>
  %vc2 = spirv.Constant dense<[-3, -15, 28]> : vector<3xi32>

  // CHECK: spirv.Constant dense<[39, -70, 155]>
  %0 = spirv.IAdd %vc1, %vc2 : vector<3xi32>
  return %0: vector<3xi32>
}

// CHECK-LABEL: @iadd_poison
//       CHECK:   %[[P:.*]] = ub.poison : i32
//       CHECK:   return %[[P]]
func.func @iadd_poison(%arg0: i32) -> i32 {
  %0 = ub.poison : i32
  %1 = spirv.IAdd %arg0, %0 : i32
  return %1: i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IAddCarry
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iaddcarry_x_0
func.func @iaddcarry_x_0(%arg0 : i32) -> !spirv.struct<(i32, i32)> {
  // CHECK: %[[RET:.*]] = spirv.CompositeConstruct
  %c0 = spirv.Constant 0 : i32
  %0 = spirv.IAddCarry %arg0, %c0 : !spirv.struct<(i32, i32)>

  // CHECK: return %[[RET]]
  return %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @const_fold_scalar_iaddcarry
func.func @const_fold_scalar_iaddcarry() -> (!spirv.struct<(i32, i32)>, !spirv.struct<(i32, i32)>) {
  %c5 = spirv.Constant 5 : i32
  %cn5 = spirv.Constant -5 : i32
  %cn8 = spirv.Constant -8 : i32

  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CN3:.*]] = spirv.Constant -3
  // CHECK-DAG: %[[UNDEF1:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER1:.*]] = spirv.CompositeInsert %[[CN3]], %[[UNDEF1]][0 : i32]
  // CHECK-DAG: %[[CC_CN3_C0:.*]] = spirv.CompositeInsert %[[C0]], %[[INTER1]][1 : i32]
  // CHECK-DAG: %[[C1:.*]] = spirv.Constant 1
  // CHECK-DAG: %[[CN13:.*]] = spirv.Constant -13
  // CHECK-DAG: %[[UNDEF2:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER2:.*]] = spirv.CompositeInsert %[[CN13]], %[[UNDEF2]][0 : i32]
  // CHECK-DAG: %[[CC_CN13_C1:.*]] = spirv.CompositeInsert %[[C1]], %[[INTER2]][1 : i32]
  %0 = spirv.IAddCarry %c5, %cn8 : !spirv.struct<(i32, i32)>
  %1 = spirv.IAddCarry %cn5, %cn8 : !spirv.struct<(i32, i32)>

  // CHECK: return %[[CC_CN3_C0]], %[[CC_CN13_C1]]
  return %0, %1 : !spirv.struct<(i32, i32)>, !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @const_fold_vector_iaddcarry
func.func @const_fold_vector_iaddcarry() -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> {
  %v0 = spirv.Constant dense<[5, -3, -1]> : vector<3xi32>
  %v1 = spirv.Constant dense<[-8, -8, 1]> : vector<3xi32>

  // CHECK-DAG: %[[CV1:.*]] = spirv.Constant dense<[-3, -11, 0]>
  // CHECK-DAG: %[[CV2:.*]] = spirv.Constant dense<[0, 1, 1]>
  // CHECK-DAG: %[[UNDEF:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER:.*]] = spirv.CompositeInsert %[[CV1]], %[[UNDEF]][0 : i32]
  // CHECK-DAG: %[[CC_CV1_CV2:.*]] = spirv.CompositeInsert %[[CV2]], %[[INTER]][1 : i32]
  %0 = spirv.IAddCarry %v0, %v1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>

  // CHECK: return %[[CC_CV1_CV2]]
  return %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @imul_zero_one
// CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @imul_zero_one(%arg0: i32) -> (i32, i32) {
  // CHECK: %[[ZERO:.*]] = spirv.Constant 0
  %zero = spirv.Constant 0 : i32
  %one = spirv.Constant 1: i32
  %0 = spirv.IMul %arg0, %zero : i32
  %1 = spirv.IMul %one, %arg0 : i32
  // CHECK: return %[[ZERO]], %[[ARG]]
  return %0, %1: i32, i32
}

// CHECK-LABEL: @const_fold_scalar_imul_normal
func.func @const_fold_scalar_imul_normal() -> (i32, i32, i32) {
  %c5 = spirv.Constant 5 : i32
  %cn8 = spirv.Constant -8 : i32
  %c7 = spirv.Constant 7 : i32

  // CHECK-DAG: spirv.Constant -56
  // CHECK-DAG: spirv.Constant -40
  // CHECK-DAG: spirv.Constant 35
  %0 = spirv.IMul %c7, %c5 : i32
  %1 = spirv.IMul %c5, %cn8 : i32
  %2 = spirv.IMul %cn8, %c7 : i32
  return %0, %1, %2: i32, i32, i32
}

// CHECK-LABEL: @const_fold_scalar_imul_flow
func.func @const_fold_scalar_imul_flow() -> (i32, i32, i32) {
  %c1 = spirv.Constant 2 : i32
  %c2 = spirv.Constant 4 : i32
  %c3 = spirv.Constant 4294967295 : i32  // 2^32 - 1 : 0xffff ffff
  %c4 = spirv.Constant 2147483647 : i32  // 2^31 - 1 : 0x7fff ffff

  // (0x7fff ffff << 2) = 0x1 ffff fffc -> 0xffff fffc
  // CHECK-DAG: %[[CST4:.*]] = spirv.Constant -4

  // (0xffff ffff << 1) = 0x1 ffff fffe -> 0xffff fffe
  // CHECK-DAG: %[[CST2:.*]] = spirv.Constant -2
  %0 = spirv.IMul %c1, %c3 : i32
  // (0x7fff ffff << 1) = 0x0 ffff fffe -> 0xffff fffe
  %1 = spirv.IMul %c1, %c4 : i32
  %2 = spirv.IMul %c4, %c2 : i32
  // CHECK: return %[[CST2]], %[[CST2]], %[[CST4]]
  return %0, %1, %2: i32, i32, i32
}


// CHECK-LABEL: @const_fold_vector_imul
func.func @const_fold_vector_imul() -> vector<3xi32> {
  %vc1 = spirv.Constant dense<[42, -55, 127]> : vector<3xi32>
  %vc2 = spirv.Constant dense<[-3, -15, 28]> : vector<3xi32>

  // CHECK: spirv.Constant dense<[-126, 825, 3556]>
  %0 = spirv.IMul %vc1, %vc2 : vector<3xi32>
  return %0: vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SMulExtended
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smulextended_x_0
func.func @smulextended_x_0(%arg0 : i32) -> !spirv.struct<(i32, i32)> {
  // CHECK: %[[C0:.*]] = spirv.Constant 0
  // CHECK: %[[RET:.*]] = spirv.CompositeConstruct %[[C0]], %[[C0]]
  %c0 = spirv.Constant 0 : i32
  %0 = spirv.SMulExtended %arg0, %c0 : !spirv.struct<(i32, i32)>

  // CHECK: return %[[RET]]
  return %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @const_fold_scalar_smulextended
func.func @const_fold_scalar_smulextended() -> (!spirv.struct<(i32, i32)>, !spirv.struct<(i32, i32)>) {
  %c5 = spirv.Constant 5 : i32
  %cn5 = spirv.Constant -5 : i32
  %cn8 = spirv.Constant -8 : i32

  // CHECK-DAG: %[[CN40:.*]] = spirv.Constant -40
  // CHECK-DAG: %[[CN1:.*]] = spirv.Constant -1
  // CHECK-DAG: %[[UNDEF1:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER1:.*]] = spirv.CompositeInsert %[[CN40]], %[[UNDEF1]][0 : i32]
  // CHECK-DAG: %[[CC_CN40_CN1:.*]] = spirv.CompositeInsert %[[CN1]], %[[INTER1]]
  // CHECK-DAG: %[[C40:.*]] = spirv.Constant 40
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[UNDEF2:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER2:.*]] = spirv.CompositeInsert %[[C40]], %[[UNDEF2]][0 : i32]
  // CHECK-DAG: %[[CC_C40_C0:.*]] = spirv.CompositeInsert %[[C0]], %[[INTER2]][1 : i32]
  %0 = spirv.SMulExtended %c5, %cn8 : !spirv.struct<(i32, i32)>
  %1 = spirv.SMulExtended %cn5, %cn8 : !spirv.struct<(i32, i32)>

  // CHECK: return %[[CC_CN40_CN1]], %[[CC_C40_C0]]
  return %0, %1 : !spirv.struct<(i32, i32)>, !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @const_fold_vector_smulextended
func.func @const_fold_vector_smulextended() -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> {
  %v0 = spirv.Constant dense<[2147483647, -5, -1]> : vector<3xi32>
  %v1 = spirv.Constant dense<[5, -8, 1]> : vector<3xi32>

  // CHECK-DAG: %[[CV1:.*]] = spirv.Constant dense<[2147483643, 40, -1]>
  // CHECK-DAG: %[[CV2:.*]] = spirv.Constant dense<[2, 0, -1]>
  // CHECK-DAG: %[[UNDEF:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER:.*]] = spirv.CompositeInsert %[[CV1]], %[[UNDEF]][0 : i32]
  // CHECK-DAG: %[[CC_CV1_CV2:.*]] = spirv.CompositeInsert %[[CV2]], %[[INTER]][1 : i32]
  %0 = spirv.SMulExtended %v0, %v1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>

  // CHECK: return %[[CC_CV1_CV2]]
  return %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>

}

// -----

//===----------------------------------------------------------------------===//
// spirv.UMulExtended
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @umulextended_x_0
func.func @umulextended_x_0(%arg0 : i32) -> !spirv.struct<(i32, i32)> {
  // CHECK: %[[C0:.*]] = spirv.Constant 0
  // CHECK: %[[RET:.*]] = spirv.CompositeConstruct %[[C0]], %[[C0]]
  %c0 = spirv.Constant 0 : i32
  %0 = spirv.UMulExtended %arg0, %c0 : !spirv.struct<(i32, i32)>

  // CHECK: return %[[RET]]
  return %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @umulextended_x_1
// CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @umulextended_x_1(%arg0 : i32) -> !spirv.struct<(i32, i32)> {
  // CHECK: %[[C0:.*]] = spirv.Constant 0
  // CHECK: %[[RET:.*]] = spirv.CompositeConstruct %[[ARG]], %[[C0]]
  %c0 = spirv.Constant 1 : i32
  %0 = spirv.UMulExtended %arg0, %c0 : !spirv.struct<(i32, i32)>

  // CHECK: return %[[RET]]
  return %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @const_fold_scalar_umulextended
func.func @const_fold_scalar_umulextended() -> (!spirv.struct<(i32, i32)>, !spirv.struct<(i32, i32)>) {
  %c5 = spirv.Constant 5 : i32
  %cn5 = spirv.Constant -5 : i32
  %cn8 = spirv.Constant -8 : i32


  // CHECK-DAG: %[[C40:.*]] = spirv.Constant 40
  // CHECK-DAG: %[[CN13:.*]] = spirv.Constant -13
  // CHECK-DAG: %[[CN40:.*]] = spirv.Constant -40
  // CHECK-DAG: %[[C4:.*]] = spirv.Constant 4
  // CHECK-DAG: %[[UNDEF1:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER1:.*]] = spirv.CompositeInsert %[[CN40]], %[[UNDEF1]][0 : i32]
  // CHECK-DAG: %[[CC_CN40_C4:.*]] = spirv.CompositeInsert %[[C4]], %[[INTER1]][1 : i32]
  // CHECK-DAG: %[[UNDEF2:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER2:.*]] = spirv.CompositeInsert %[[C40]], %[[UNDEF2]][0 : i32]
  // CHECK-DAG: %[[CC_C40_CN13:.*]] = spirv.CompositeInsert %[[CN13]], %[[INTER2]][1 : i32]
  %0 = spirv.UMulExtended %c5, %cn8 : !spirv.struct<(i32, i32)>
  %1 = spirv.UMulExtended %cn5, %cn8 : !spirv.struct<(i32, i32)>

  // CHECK: return %[[CC_CN40_C4]], %[[CC_C40_CN13]]
  return %0, %1 : !spirv.struct<(i32, i32)>, !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @const_fold_vector_umulextended
func.func @const_fold_vector_umulextended() -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> {
  %v0 = spirv.Constant dense<[2147483647, -5, -1]> : vector<3xi32>
  %v1 = spirv.Constant dense<[5, -8, 1]> : vector<3xi32>

  // CHECK-DAG: %[[CV1:.*]] = spirv.Constant dense<[2147483643, 40, -1]>
  // CHECK-DAG: %[[CV2:.*]] = spirv.Constant dense<[2, -13, 0]>
  // CHECK-DAG: %[[UNDEF:.*]] = spirv.Undef
  // CHECK-DAG: %[[INTER:.*]] = spirv.CompositeInsert %[[CV1]], %[[UNDEF]]
  // CHECK-DAG: %[[CC_CV1_CV2:.*]] = spirv.CompositeInsert %[[CV2]], %[[INTER]]
  %0 = spirv.UMulExtended %v0, %v1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>

  // CHECK: return %[[CC_CV1_CV2]]
  return %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// -----


//===----------------------------------------------------------------------===//
// spirv.ISub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @isub_x_x
func.func @isub_x_x(%arg0: i32) -> i32 {
  // CHECK: spirv.Constant 0
  %0 = spirv.ISub %arg0, %arg0: i32
  return %0: i32
}

// CHECK-LABEL: @const_fold_scalar_isub_normal
func.func @const_fold_scalar_isub_normal() -> (i32, i32, i32) {
  %c5 = spirv.Constant 5 : i32
  %cn8 = spirv.Constant -8 : i32
  %c7 = spirv.Constant 7 : i32

  // CHECK-DAG: spirv.Constant -15
  // CHECK-DAG: spirv.Constant 13
  // CHECK-DAG: spirv.Constant 2
  %0 = spirv.ISub %c7, %c5 : i32
  %1 = spirv.ISub %c5, %cn8 : i32
  %2 = spirv.ISub %cn8, %c7 : i32
  return %0, %1, %2: i32, i32, i32
}

// CHECK-LABEL: @const_fold_scalar_isub_flow
func.func @const_fold_scalar_isub_flow() -> (i32, i32, i32, i32) {
  %c1 = spirv.Constant 0 : i32
  %c2 = spirv.Constant 1 : i32
  %c3 = spirv.Constant 4294967295 : i32  // 2^32 - 1 : 0xffff ffff
  %c4 = spirv.Constant 2147483647 : i32  // 2^31     : 0x7fff ffff
  %c5 = spirv.Constant -1 : i32          //          : 0xffff ffff
  %c6 = spirv.Constant -2 : i32          //          : 0xffff fffe

  // 0xffff ffff - 0x7fff ffff -> 0xffff ffff + 0x8000 0001 = 0x1 8000 0000
  // CHECK-DAG: spirv.Constant -2147483648
  // 0x0000 0001 - 0xffff ffff -> 0x0000 0001 + 0x0000 0001 = 0x0000 0002
  // CHECK-DAG: spirv.Constant 2 :
  // 0x0000 0000 - 0xffff ffff -> 0x0000 0000 + 0x0000 0001 = 0x0000 0001
  // CHECK-DAG: spirv.Constant 1 :
  // 0xffff fffe - 0x7fff ffff -> 0xffff fffe + 0x8000 0001 = 0x1 7fff ffff
  // CHECK-DAG: spirv.Constant 2147483647
  %0 = spirv.ISub %c1, %c3 : i32
  %1 = spirv.ISub %c2, %c3 : i32
  %2 = spirv.ISub %c5, %c4 : i32
  %3 = spirv.ISub %c6, %c4 : i32
  return %0, %1, %2, %3: i32, i32, i32, i32
}

// CHECK-LABEL: @const_fold_vector_isub
func.func @const_fold_vector_isub() -> vector<3xi32> {
  %vc1 = spirv.Constant dense<[42, -55, 127]> : vector<3xi32>
  %vc2 = spirv.Constant dense<[-3, -15, 28]> : vector<3xi32>

  // CHECK: spirv.Constant dense<[45, -40, 99]>
  %0 = spirv.ISub %vc1, %vc2 : vector<3xi32>
  return %0: vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sdiv_x_1
func.func @sdiv_x_1(%arg0 : i32) -> i32 {
  // CHECK-NEXT: return %arg0 : i32
  %c1 = spirv.Constant 1  : i32
  %2 = spirv.SDiv %arg0, %c1: i32
  return %2 : i32
}

// CHECK-LABEL: @sdiv_div_0_or_overflow
func.func @sdiv_div_0_or_overflow() -> (i32, i32) {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CN1:.*]] = spirv.Constant -1
  // CHECK-DAG: %[[CNMIN:.*]] = spirv.Constant -2147483648

  %c0 = spirv.Constant 0 : i32
  %cn1 = spirv.Constant -1 : i32
  %min_i32 = spirv.Constant -2147483648 : i32

  // CHECK: %0 = spirv.SDiv %[[CN1]], %[[C0]]
  // CHECK: %1 = spirv.SDiv %[[CNMIN]], %[[CN1]]
  %0 = spirv.SDiv %cn1, %c0 : i32
  %1 = spirv.SDiv %min_i32, %cn1 : i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: @const_fold_scalar_sdiv
func.func @const_fold_scalar_sdiv() -> (i32, i32, i32, i32) {
  %c56 = spirv.Constant 56 : i32
  %c7 = spirv.Constant 7 : i32
  %cn8 = spirv.Constant -8 : i32
  %c3 = spirv.Constant 3 : i32
  %cn3 = spirv.Constant -3 : i32

  // CHECK-DAG: %[[CN18:.*]] = spirv.Constant -18
  // CHECK-DAG: %[[CN2:.*]] = spirv.Constant -2
  // CHECK-DAG: %[[CN7:.*]] = spirv.Constant -7
  // CHECK-DAG: %[[C8:.*]] = spirv.Constant 8
  %0 = spirv.SDiv %c56, %c7 : i32
  %1 = spirv.SDiv %c56, %cn8 : i32
  %2 = spirv.SDiv %cn8, %c3 : i32
  %3 = spirv.SDiv %c56, %cn3 : i32

  // CHECK: return %[[C8]], %[[CN7]], %[[CN2]], %[[CN18]]
  return %0, %1, %2, %3: i32, i32, i32, i32
}

// CHECK-LABEL: @const_fold_vector_sdiv
func.func @const_fold_vector_sdiv() -> vector<3xi32> {
  // CHECK: %[[CVEC:.*]] = spirv.Constant dense<[0, -1, -3]>

  %cv_num = spirv.Constant dense<[42, 24, -16]> : vector<3xi32>
  %cv_denom = spirv.Constant dense<[76, -24, 5]> : vector<3xi32>
  %0 = spirv.SDiv %cv_num, %cv_denom : vector<3xi32>

  // CHECK: return %[[CVEC]]
  return %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SMod
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smod_x_1
func.func @smod_x_1(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CVEC0:.*]] = spirv.Constant dense<0>
  %c1 = spirv.Constant 1 : i32
  %cv1 = spirv.Constant dense<1> : vector<3xi32>
  %0 = spirv.SMod %arg0, %c1: i32
  %1 = spirv.SMod %arg1, %cv1: vector<3xi32>

  // CHECK: return %[[C0]], %[[CVEC0]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @smod_div_0_or_overflow
func.func @smod_div_0_or_overflow() -> (i32, i32) {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CN1:.*]] = spirv.Constant -1
  // CHECK-DAG: %[[CNMIN:.*]] = spirv.Constant -2147483648

  %c0 = spirv.Constant 0 : i32
  %cn1 = spirv.Constant -1 : i32
  %min_i32 = spirv.Constant -2147483648 : i32

  // CHECK: %0 = spirv.SMod %[[CN1]], %[[C0]]
  // CHECK: %1 = spirv.SMod %[[CNMIN]], %[[CN1]]
  %0 = spirv.SMod %cn1, %c0 : i32
  %1 = spirv.SMod %min_i32, %cn1 : i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: @const_fold_scalar_smod
func.func @const_fold_scalar_smod() -> (i32, i32, i32, i32, i32, i32, i32, i32) {
  %c56 = spirv.Constant 56 : i32
  %cn56 = spirv.Constant -56 : i32
  %c59 = spirv.Constant 59 : i32
  %cn59 = spirv.Constant -59 : i32
  %c7 = spirv.Constant 7 : i32
  %cn8 = spirv.Constant -8 : i32
  %c3 = spirv.Constant 3 : i32
  %cn3 = spirv.Constant -3 : i32

  // CHECK-DAG: %[[ZERO:.*]] = spirv.Constant 0 : i32
  // CHECK-DAG: %[[TWO:.*]] = spirv.Constant 2 : i32
  // CHECK-DAG: %[[FIFTYTHREE:.*]] = spirv.Constant 53 : i32
  // CHECK-DAG: %[[NFIFTYTHREE:.*]] = spirv.Constant -53 : i32
  // CHECK-DAG: %[[THREE:.*]] = spirv.Constant 3 : i32
  // CHECK-DAG: %[[NTHREE:.*]] = spirv.Constant -3 : i32
  %0 = spirv.SMod %c56, %c7 : i32
  %1 = spirv.SMod %c56, %cn8 : i32
  %2 = spirv.SMod %c56, %c3 : i32
  %3 = spirv.SMod %cn3, %c56 : i32
  %4 = spirv.SMod %cn3, %cn56 : i32
  %5 = spirv.SMod %c59, %c56 : i32
  %6 = spirv.SMod %c59, %cn56 : i32
  %7 = spirv.SMod %cn59, %cn56 : i32

  // CHECK: return %[[ZERO]], %[[ZERO]], %[[TWO]], %[[FIFTYTHREE]], %[[NTHREE]], %[[THREE]], %[[NFIFTYTHREE]], %[[NTHREE]]
  return %0, %1, %2, %3, %4, %5, %6, %7 : i32, i32, i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: @const_fold_vector_smod
func.func @const_fold_vector_smod() -> vector<3xi32> {
  // CHECK: %[[CVEC:.*]] = spirv.Constant dense<[42, -4, 4]>

  %cv = spirv.Constant dense<[42, 24, -16]> : vector<3xi32>
  %cv_mod = spirv.Constant dense<[76, -7, 5]> : vector<3xi32>
  %0 = spirv.SMod %cv, %cv_mod : vector<3xi32>

  // CHECK: return %[[CVEC]]
  return %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SRem
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @srem_x_1
func.func @srem_x_1(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CVEC0:.*]] = spirv.Constant dense<0>
  %c1 = spirv.Constant 1 : i32
  %cv1 = spirv.Constant dense<1> : vector<3xi32>
  %0 = spirv.SRem %arg0, %c1: i32
  %1 = spirv.SRem %arg1, %cv1: vector<3xi32>

  // CHECK: return %[[C0]], %[[CVEC0]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @srem_div_0_or_overflow
func.func @srem_div_0_or_overflow() -> (i32, i32) {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CN1:.*]] = spirv.Constant -1
  // CHECK-DAG: %[[CNMIN:.*]] = spirv.Constant -2147483648
  %c0 = spirv.Constant 0 : i32
  %cn1 = spirv.Constant -1 : i32
  %min_i32 = spirv.Constant -2147483648 : i32

  // CHECK: %0 = spirv.SRem %[[CN1]], %[[C0]]
  // CHECK: %1 = spirv.SRem %[[CNMIN]], %[[CN1]]
  %0 = spirv.SRem %cn1, %c0 : i32
  %1 = spirv.SRem %min_i32, %cn1 : i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: @const_fold_scalar_srem
func.func @const_fold_scalar_srem() -> (i32, i32, i32, i32, i32) {
  %c56 = spirv.Constant 56 : i32
  %c7 = spirv.Constant 7 : i32
  %cn8 = spirv.Constant -8 : i32
  %c3 = spirv.Constant 3 : i32
  %cn3 = spirv.Constant -3 : i32

  // CHECK-DAG: %[[ONE:.*]] = spirv.Constant 1 : i32
  // CHECK-DAG: %[[NTHREE:.*]] = spirv.Constant -3 : i32
  // CHECK-DAG: %[[TWO:.*]] = spirv.Constant 2 : i32
  // CHECK-DAG: %[[ZERO:.*]] = spirv.Constant 0 : i32
  %0 = spirv.SRem %c56, %c7 : i32
  %1 = spirv.SRem %c56, %cn8 : i32
  %2 = spirv.SRem %c56, %c3 : i32
  %3 = spirv.SRem %cn3, %c56 : i32
  %4 = spirv.SRem %c7, %cn3 : i32
  // CHECK: return %[[ZERO]], %[[ZERO]], %[[TWO]], %[[NTHREE]], %[[ONE]]
  return %0, %1, %2, %3, %4 : i32, i32, i32, i32, i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @udiv_x_1
func.func @udiv_x_1(%arg0 : i32) -> i32 {
  // CHECK-NEXT: return %arg0 : i32
  %c1 = spirv.Constant 1  : i32
  %2 = spirv.UDiv %arg0, %c1: i32
  return %2 : i32
}

// CHECK-LABEL: @udiv_div_0
func.func @udiv_div_0() -> i32 {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CN1:.*]] = spirv.Constant -1
  %c0 = spirv.Constant 0 : i32
  %cn1 = spirv.Constant -1 : i32

  // CHECK: %0 = spirv.UDiv %[[CN1]], %[[C0]]
  %0 = spirv.UDiv %cn1, %c0 : i32
  return %0 : i32
}

// CHECK-LABEL: @const_fold_scalar_udiv
func.func @const_fold_scalar_udiv() -> (i32, i32, i32) {
  %c56 = spirv.Constant 56 : i32
  %c7 = spirv.Constant 7 : i32
  %cn8 = spirv.Constant -8 : i32
  %c3 = spirv.Constant 3 : i32

  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CBIG:.*]] = spirv.Constant 1431655762
  // CHECK-DAG: %[[C8:.*]] = spirv.Constant 8
  %0 = spirv.UDiv %c56, %c7 : i32
  %1 = spirv.UDiv %cn8, %c3 : i32
  %2 = spirv.UDiv %c56, %cn8 : i32

  // CHECK: return %[[C8]], %[[CBIG]], %[[C0]]
  return %0, %1, %2 : i32, i32, i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UMod
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @umod_x_1
func.func @umod_x_1(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CVEC0:.*]] = spirv.Constant dense<0>
  %c1 = spirv.Constant 1 : i32
  %cv1 = spirv.Constant dense<1> : vector<3xi32>
  %0 = spirv.UMod %arg0, %c1: i32
  %1 = spirv.UMod %arg1, %cv1: vector<3xi32>

  // CHECK: return %[[C0]], %[[CVEC0]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @umod_div_0
func.func @umod_div_0() -> i32 {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CN1:.*]] = spirv.Constant -1
  %c0 = spirv.Constant 0 : i32
  %cn1 = spirv.Constant -1 : i32

  // CHECK: %0 = spirv.UMod %[[CN1]], %[[C0]]
  %0 = spirv.UMod %cn1, %c0 : i32
  return %0 : i32
}

// CHECK-LABEL: @const_fold_scalar_umod
func.func @const_fold_scalar_umod() -> (i32, i32, i32) {
  %c56 = spirv.Constant 56 : i32
  %c7 = spirv.Constant 7 : i32
  %cn8 = spirv.Constant -8 : i32
  %c3 = spirv.Constant 3 : i32

  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[C2:.*]] = spirv.Constant 2
  // CHECK-DAG: %[[C56:.*]] = spirv.Constant 56
  %0 = spirv.UMod %c56, %c7 : i32
  %1 = spirv.UMod %cn8, %c3 : i32
  %2 = spirv.UMod %c56, %cn8 : i32

  // CHECK: return %[[C0]], %[[C2]], %[[C56]]
  return %0, %1, %2 : i32, i32, i32
}

// CHECK-LABEL: @const_fold_vector_umod
func.func @const_fold_vector_umod() -> vector<3xi32> {
  // CHECK: %[[CVEC:.*]] = spirv.Constant dense<[42, 24, 0]>

  %cv = spirv.Constant dense<[42, 24, -16]> : vector<3xi32>
  %cv_mod = spirv.Constant dense<[76, -7, 5]> : vector<3xi32>
  %0 = spirv.UMod %cv, %cv_mod : vector<3xi32>

  // CHECK: return %[[CVEC]]
  return %0 : vector<3xi32>
}

// CHECK-LABEL: @umod_fold
// CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @umod_fold(%arg0: i32) -> (i32, i32) {
  // CHECK: %[[CONST4:.*]] = spirv.Constant 4
  // CHECK: %[[CONST32:.*]] = spirv.Constant 32
  %const1 = spirv.Constant 32 : i32
  %0 = spirv.UMod %arg0, %const1 : i32
  %const2 = spirv.Constant 4 : i32
  %1 = spirv.UMod %0, %const2 : i32
  // CHECK: %[[UMOD0:.*]] = spirv.UMod %[[ARG]], %[[CONST32]]
  // CHECK: %[[UMOD1:.*]] = spirv.UMod %[[ARG]], %[[CONST4]]
  // CHECK: return %[[UMOD0]], %[[UMOD1]]
  return %0, %1: i32, i32
}

// CHECK-LABEL: @umod_fail_vector_fold
// CHECK-SAME: (%[[ARG:.*]]: vector<4xi32>)
func.func @umod_fail_vector_fold(%arg0: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  // CHECK: %[[CONST4:.*]] = spirv.Constant dense<4> : vector<4xi32>
  // CHECK: %[[CONST32:.*]] = spirv.Constant dense<32> : vector<4xi32>
  %const1 = spirv.Constant dense<32> : vector<4xi32>
  %0 = spirv.UMod %arg0, %const1 : vector<4xi32>
  // CHECK: %[[UMOD0:.*]] = spirv.UMod %[[ARG]], %[[CONST32]]
  %const2 = spirv.Constant dense<4> : vector<4xi32>
  %1 = spirv.UMod %0, %const2 : vector<4xi32>
  // CHECK: %[[UMOD1:.*]] = spirv.UMod %[[UMOD0]], %[[CONST4]]
  // CHECK: return %[[UMOD0]], %[[UMOD1]]
  return %0, %1: vector<4xi32>, vector<4xi32>
} 

// CHECK-LABEL: @umod_fold_same_divisor
// CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @umod_fold_same_divisor(%arg0: i32) -> (i32, i32) {
  // CHECK: %[[CONST1:.*]] = spirv.Constant 32
  %const1 = spirv.Constant 32 : i32
  %0 = spirv.UMod %arg0, %const1 : i32
  %const2 = spirv.Constant 32 : i32
  %1 = spirv.UMod %0, %const2 : i32
  // CHECK: %[[UMOD0:.*]] = spirv.UMod %[[ARG]], %[[CONST1]]
  // CHECK: %[[UMOD1:.*]] = spirv.UMod %[[ARG]], %[[CONST1]]
  // CHECK: return %[[UMOD0]], %[[UMOD1]]
  return %0, %1: i32, i32
}

// CHECK-LABEL: @umod_fail_fold
// CHECK-SAME: (%[[ARG:.*]]: i32)
func.func @umod_fail_fold(%arg0: i32) -> (i32, i32) {
  // CHECK: %[[CONST5:.*]] = spirv.Constant 5
  // CHECK: %[[CONST32:.*]] = spirv.Constant 32
  %const1 = spirv.Constant 32 : i32
  %0 = spirv.UMod %arg0, %const1 : i32
  // CHECK: %[[UMOD0:.*]] = spirv.UMod %[[ARG]], %[[CONST32]]
  %const2 = spirv.Constant 5 : i32
  %1 = spirv.UMod %0, %const2 : i32
  // CHECK: %[[UMOD1:.*]] = spirv.UMod %[[UMOD0]], %[[CONST5]]
  // CHECK: return %[[UMOD0]], %[[UMOD1]]
  return %0, %1: i32, i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.LogicalAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_logical_and_true_false_scalar
// CHECK-SAME: %[[ARG:.+]]: i1
func.func @convert_logical_and_true_false_scalar(%arg: i1) -> (i1, i1) {
  %true = spirv.Constant true
  // CHECK: %[[FALSE:.+]] = spirv.Constant false
  %false = spirv.Constant false
  %0 = spirv.LogicalAnd %true, %arg: i1
  %1 = spirv.LogicalAnd %arg, %false: i1
  // CHECK: return %[[ARG]], %[[FALSE]]
  return %0, %1: i1, i1
}

// CHECK-LABEL: @convert_logical_and_true_false_vector
// CHECK-SAME: %[[ARG:.+]]: vector<3xi1>
func.func @convert_logical_and_true_false_vector(%arg: vector<3xi1>) -> (vector<3xi1>, vector<3xi1>) {
  %true = spirv.Constant dense<true> : vector<3xi1>
  // CHECK: %[[FALSE:.+]] = spirv.Constant dense<false>
  %false = spirv.Constant dense<false> : vector<3xi1>
  %0 = spirv.LogicalAnd %true, %arg: vector<3xi1>
  %1 = spirv.LogicalAnd %arg, %false: vector<3xi1>
  // CHECK: return %[[ARG]], %[[FALSE]]
  return %0, %1: vector<3xi1>, vector<3xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.LogicalNot
//===----------------------------------------------------------------------===//

func.func @convert_logical_not_to_not_equal(%arg0: vector<3xi64>, %arg1: vector<3xi64>) -> vector<3xi1> {
  // CHECK: %[[RESULT:.*]] = spirv.INotEqual {{%.*}}, {{%.*}} : vector<3xi64>
  // CHECK-NEXT: spirv.ReturnValue %[[RESULT]] : vector<3xi1>
  %2 = spirv.IEqual %arg0, %arg1 : vector<3xi64>
  %3 = spirv.LogicalNot %2 : vector<3xi1>
  spirv.ReturnValue %3 : vector<3xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.LogicalEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_equal_same
func.func @logical_equal_same(%arg0 : i1, %arg1 : vector<3xi1>) -> (i1, vector<3xi1>) {
  // CHECK-DAG: %[[CTRUE:.*]] = spirv.Constant true
  // CHECK-DAG: %[[CVTRUE:.*]] = spirv.Constant dense<true>

  %0 = spirv.LogicalEqual %arg0, %arg0 : i1
  %1 = spirv.LogicalEqual %arg1, %arg1 : vector<3xi1>
  // CHECK: return %[[CTRUE]], %[[CVTRUE]]
  return %0, %1 : i1, vector<3xi1>
}

// CHECK-LABEL: @const_fold_scalar_logical_equal
func.func @const_fold_scalar_logical_equal() -> (i1, i1) {
  %true = spirv.Constant true
  %false = spirv.Constant false

  // CHECK-DAG: %[[CTRUE:.*]] = spirv.Constant true
  // CHECK-DAG: %[[CFALSE:.*]] = spirv.Constant false
  %0 = spirv.LogicalEqual %true, %false : i1
  %1 = spirv.LogicalEqual %false, %false : i1

  // CHECK: return %[[CFALSE]], %[[CTRUE]]
  return %0, %1 : i1, i1
}

// CHECK-LABEL: @const_fold_vector_logical_equal
func.func @const_fold_vector_logical_equal() -> vector<3xi1> {
  %cv0 = spirv.Constant dense<[true, false, true]> : vector<3xi1>
  %cv1 = spirv.Constant dense<[true, false, false]> : vector<3xi1>

  // CHECK: %[[RET:.*]] = spirv.Constant dense<[true, true, false]>
  %0 = spirv.LogicalEqual %cv0, %cv1 : vector<3xi1>

  // CHECK: return %[[RET]]
  return %0 : vector<3xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.LogicalNotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_logical_not_equal_false
// CHECK-SAME: %[[ARG:.+]]: vector<4xi1>
func.func @convert_logical_not_equal_false(%arg: vector<4xi1>) -> vector<4xi1> {
  %cst = spirv.Constant dense<false> : vector<4xi1>
  // CHECK: spirv.ReturnValue %[[ARG]] : vector<4xi1>
  %0 = spirv.LogicalNotEqual %arg, %cst : vector<4xi1>
  spirv.ReturnValue %0 : vector<4xi1>
}

// CHECK-LABEL: @logical_not_equal_same
func.func @logical_not_equal_same(%arg0 : i1, %arg1 : vector<3xi1>) -> (i1, vector<3xi1>) {
  // CHECK-DAG: %[[CFALSE:.*]] = spirv.Constant false
  // CHECK-DAG: %[[CVFALSE:.*]] = spirv.Constant dense<false>
  %0 = spirv.LogicalNotEqual %arg0, %arg0 : i1
  %1 = spirv.LogicalNotEqual %arg1, %arg1 : vector<3xi1>

  // CHECK: return %[[CFALSE]], %[[CVFALSE]]
  return %0, %1 : i1, vector<3xi1>
}

// CHECK-LABEL: @const_fold_scalar_logical_not_equal
func.func @const_fold_scalar_logical_not_equal() -> (i1, i1) {
  %true = spirv.Constant true
  %false = spirv.Constant false

  // CHECK-DAG: %[[CTRUE:.*]] = spirv.Constant true
  // CHECK-DAG: %[[CFALSE:.*]] = spirv.Constant false
  %0 = spirv.LogicalNotEqual %true, %false : i1
  %1 = spirv.LogicalNotEqual %false, %false : i1

  // CHECK: return %[[CTRUE]], %[[CFALSE]]
  return %0, %1 : i1, i1
}

// CHECK-LABEL: @const_fold_vector_logical_not_equal
func.func @const_fold_vector_logical_not_equal() -> vector<3xi1> {
  %cv0 = spirv.Constant dense<[true, false, true]> : vector<3xi1>
  %cv1 = spirv.Constant dense<[true, false, false]> : vector<3xi1>

  // CHECK: %[[RET:.*]] = spirv.Constant dense<[false, false, true]>
  %0 = spirv.LogicalNotEqual %cv0, %cv1 : vector<3xi1>

  // CHECK: return %[[RET]]
  return %0 : vector<3xi1>
}

// -----

func.func @convert_logical_not_to_equal(%arg0: vector<3xi64>, %arg1: vector<3xi64>) -> vector<3xi1> {
  // CHECK: %[[RESULT:.*]] = spirv.IEqual {{%.*}}, {{%.*}} : vector<3xi64>
  // CHECK-NEXT: spirv.ReturnValue %[[RESULT]] : vector<3xi1>
  %2 = spirv.INotEqual %arg0, %arg1 : vector<3xi64>
  %3 = spirv.LogicalNot %2 : vector<3xi1>
  spirv.ReturnValue %3 : vector<3xi1>
}

// -----

func.func @convert_logical_not_parent_multi_use(%arg0: vector<3xi64>, %arg1: vector<3xi64>, %arg2: !spirv.ptr<vector<3xi1>, Uniform>) -> vector<3xi1> {
  // CHECK: %[[RESULT_0:.*]] = spirv.INotEqual {{%.*}}, {{%.*}} : vector<3xi64>
  // CHECK-NEXT: %[[RESULT_1:.*]] = spirv.IEqual {{%.*}}, {{%.*}} : vector<3xi64>
  // CHECK-NEXT: spirv.Store "Uniform" {{%.*}}, %[[RESULT_0]]
  // CHECK-NEXT: spirv.ReturnValue %[[RESULT_1]]
  %0 = spirv.INotEqual %arg0, %arg1 : vector<3xi64>
  %1 = spirv.LogicalNot %0 : vector<3xi1>
  spirv.Store "Uniform" %arg2, %0 : vector<3xi1>
  spirv.ReturnValue %1 : vector<3xi1>
}

// -----

func.func @convert_logical_not_to_logical_not_equal(%arg0: vector<3xi1>, %arg1: vector<3xi1>) -> vector<3xi1> {
  // CHECK: %[[RESULT:.*]] = spirv.LogicalNotEqual {{%.*}}, {{%.*}} : vector<3xi1>
  // CHECK-NEXT: spirv.ReturnValue %[[RESULT]] : vector<3xi1>
  %2 = spirv.LogicalEqual %arg0, %arg1 : vector<3xi1>
  %3 = spirv.LogicalNot %2 : vector<3xi1>
  spirv.ReturnValue %3 : vector<3xi1>
}

// -----

func.func @convert_logical_not_to_logical_equal(%arg0: vector<3xi1>, %arg1: vector<3xi1>) -> vector<3xi1> {
  // CHECK: %[[RESULT:.*]] = spirv.LogicalEqual {{%.*}}, {{%.*}} : vector<3xi1>
  // CHECK-NEXT: spirv.ReturnValue %[[RESULT]] : vector<3xi1>
  %2 = spirv.LogicalNotEqual %arg0, %arg1 : vector<3xi1>
  %3 = spirv.LogicalNot %2 : vector<3xi1>
  spirv.ReturnValue %3 : vector<3xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.LogicalOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @convert_logical_or_true_false_scalar
// CHECK-SAME: %[[ARG:.+]]: i1
func.func @convert_logical_or_true_false_scalar(%arg: i1) -> (i1, i1) {
  // CHECK: %[[TRUE:.+]] = spirv.Constant true
  %true = spirv.Constant true
  %false = spirv.Constant false
  %0 = spirv.LogicalOr %true, %arg: i1
  %1 = spirv.LogicalOr %arg, %false: i1
  // CHECK: return %[[TRUE]], %[[ARG]]
  return %0, %1: i1, i1
}

// CHECK-LABEL: @convert_logical_or_true_false_vector
// CHECK-SAME: %[[ARG:.+]]: vector<3xi1>
func.func @convert_logical_or_true_false_vector(%arg: vector<3xi1>) -> (vector<3xi1>, vector<3xi1>) {
  // CHECK: %[[TRUE:.+]] = spirv.Constant dense<true>
  %true = spirv.Constant dense<true> : vector<3xi1>
  %false = spirv.Constant dense<false> : vector<3xi1>
  %0 = spirv.LogicalOr %true, %arg: vector<3xi1>
  %1 = spirv.LogicalOr %arg, %false: vector<3xi1>
  // CHECK: return %[[TRUE]], %[[ARG]]
  return %0, %1: vector<3xi1>, vector<3xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iequal_same
func.func @iequal_same(%arg0 : i32, %arg1 : vector<3xi32>) -> (i1, vector<3xi1>) {
  // CHECK-DAG: %[[CTRUE:.*]] = spirv.Constant true
  // CHECK-DAG: %[[CVTRUE:.*]] = spirv.Constant dense<true>
  %0 = spirv.IEqual %arg0, %arg0 : i32
  %1 = spirv.IEqual %arg1, %arg1 : vector<3xi32>

  // CHECK: return %[[CTRUE]], %[[CVTRUE]]
  return %0, %1 : i1, vector<3xi1>
}

// CHECK-LABEL: @const_fold_scalar_iequal
func.func @const_fold_scalar_iequal() -> (i1, i1) {
  %c5 = spirv.Constant 5 : i32
  %c6 = spirv.Constant 6 : i32

  // CHECK-DAG: %[[CTRUE:.*]] = spirv.Constant true
  // CHECK-DAG: %[[CFALSE:.*]] = spirv.Constant false
  %0 = spirv.IEqual %c5, %c6 : i32
  %1 = spirv.IEqual %c5, %c5 : i32

  // CHECK: return %[[CFALSE]], %[[CTRUE]]
  return %0, %1 : i1, i1
}

// CHECK-LABEL: @const_fold_vector_iequal
func.func @const_fold_vector_iequal() -> vector<3xi1> {
  %cv0 = spirv.Constant dense<[-1, -4, 2]> : vector<3xi32>
  %cv1 = spirv.Constant dense<[-1, -3, 2]> : vector<3xi32>

  // CHECK: %[[RET:.*]] = spirv.Constant dense<[true, false, true]>
  %0 = spirv.IEqual %cv0, %cv1 : vector<3xi32>

  // CHECK: return %[[RET]]
  return %0 : vector<3xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.INotEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @inotequal_same
func.func @inotequal_same(%arg0 : i32, %arg1 : vector<3xi32>) -> (i1, vector<3xi1>) {
  // CHECK-DAG: %[[CFALSE:.*]] = spirv.Constant false
  // CHECK-DAG: %[[CVFALSE:.*]] = spirv.Constant dense<false>
  %0 = spirv.INotEqual %arg0, %arg0 : i32
  %1 = spirv.INotEqual %arg1, %arg1 : vector<3xi32>

  // CHECK: return %[[CFALSE]], %[[CVFALSE]]
  return %0, %1 : i1, vector<3xi1>
}

// CHECK-LABEL: @const_fold_scalar_inotequal
func.func @const_fold_scalar_inotequal() -> (i1, i1) {
  %c5 = spirv.Constant 5 : i32
  %c6 = spirv.Constant 6 : i32

  // CHECK-DAG: %[[CTRUE:.*]] = spirv.Constant true
  // CHECK-DAG: %[[CFALSE:.*]] = spirv.Constant false
  %0 = spirv.INotEqual %c5, %c6 : i32
  %1 = spirv.INotEqual %c5, %c5 : i32

  // CHECK: return %[[CTRUE]], %[[CFALSE]]
  return %0, %1 : i1, i1
}

// CHECK-LABEL: @const_fold_vector_inotequal
func.func @const_fold_vector_inotequal() -> vector<3xi1> {
  %cv0 = spirv.Constant dense<[-1, -4, 2]> : vector<3xi32>
  %cv1 = spirv.Constant dense<[-1, -3, 2]> : vector<3xi32>

  // CHECK: %[[RET:.*]] = spirv.Constant dense<[false, true, false]>
  %0 = spirv.INotEqual %cv0, %cv1 : vector<3xi32>

  // CHECK: return %[[RET]]
  return %0 : vector<3xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.LeftShiftLogical
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @lsl_x_0
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @lsl_x_0(%arg0 : i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  %c0 = spirv.Constant 0 : i32
  %cv0 = spirv.Constant dense<0> : vector<3xi32>

  %0 = spirv.ShiftLeftLogical %arg0, %c0 : i32, i32
  %1 = spirv.ShiftLeftLogical %arg1, %cv0 : vector<3xi32>, vector<3xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @lsl_shift_overflow
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @lsl_shift_overflow(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[C32:.*]] = spirv.Constant 32
  // CHECK-DAG: %[[CV:.*]] = spirv.Constant dense<[6, 18, 128]>
  %c32 = spirv.Constant 32 : i32
  %cv = spirv.Constant dense<[6, 18, 128]> : vector<3xi32>

  // CHECK: %0 = spirv.ShiftLeftLogical %[[ARG0]], %[[C32]]
  // CHECK: %1 = spirv.ShiftLeftLogical %[[ARG1]], %[[CV]]
  %0 = spirv.ShiftLeftLogical %arg0, %c32 : i32, i32
  %1 = spirv.ShiftLeftLogical %arg1, %cv : vector<3xi32>, vector<3xi32>

  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @const_fold_scalar_lsl
func.func @const_fold_scalar_lsl() -> i32 {
  %c1 = spirv.Constant 65535 : i32  // 0x0000 ffff
  %c2 = spirv.Constant 17 : i32

  // CHECK: %[[RET:.*]] = spirv.Constant -131072
  // 0x0000 ffff << 17 -> 0xfffe 0000
  %0 = spirv.ShiftLeftLogical %c1, %c2 : i32, i32

  // CHECK: return %[[RET]]
  return %0 : i32
}

// CHECK-LABEL: @const_fold_vector_lsl
func.func @const_fold_vector_lsl() -> vector<3xi32> {
  %c1 = spirv.Constant dense<[1, -1, 127]> : vector<3xi32>
  %c2 = spirv.Constant dense<[31, 16, 13]> : vector<3xi32>

  // CHECK: %[[RET:.*]] = spirv.Constant dense<[-2147483648, -65536, 1040384]>
  %0 = spirv.ShiftLeftLogical %c1, %c2 : vector<3xi32>, vector<3xi32>

  // CHECK: return %[[RET]]
  return %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.RightShiftArithmetic
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @asr_x_0
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @asr_x_0(%arg0 : i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  %c0 = spirv.Constant 0 : i32
  %cv0 = spirv.Constant dense<0> : vector<3xi32>

  %0 = spirv.ShiftRightArithmetic %arg0, %c0 : i32, i32
  %1 = spirv.ShiftRightArithmetic %arg1, %cv0 : vector<3xi32>, vector<3xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @asr_shift_overflow
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @asr_shift_overflow(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[C32:.*]] = spirv.Constant 32
  // CHECK-DAG: %[[CV:.*]] = spirv.Constant dense<[6, 18, 128]>
  %c32 = spirv.Constant 32 : i32
  %cv = spirv.Constant dense<[6, 18, 128]> : vector<3xi32>

  // CHECK: %0 = spirv.ShiftRightArithmetic %[[ARG0]], %[[C32]]
  // CHECK: %1 = spirv.ShiftRightArithmetic %[[ARG1]], %[[CV]]
  %0 = spirv.ShiftRightArithmetic %arg0, %c32 : i32, i32
  %1 = spirv.ShiftRightArithmetic %arg1, %cv : vector<3xi32>, vector<3xi32>

  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @const_fold_scalar_asr
func.func @const_fold_scalar_asr() -> i32 {
  %c1 = spirv.Constant -131072 : i32  // 0xfffe 0000
  %c2 = spirv.Constant 17 : i32
  // 0x0000 ffff ashr 17 -> 0xffff ffff
  // CHECK: %[[RET:.*]] = spirv.Constant -1
  %0 = spirv.ShiftRightArithmetic %c1, %c2 : i32, i32

  // CHECK: return %[[RET]]
  return %0 : i32
}

// CHECK-LABEL: @const_fold_vector_asr
func.func @const_fold_vector_asr() -> vector<3xi32> {
  %c1 = spirv.Constant dense<[-2147483648, 239847, 127]> : vector<3xi32>
  %c2 = spirv.Constant dense<[31, 16, 13]> : vector<3xi32>

  // CHECK: %[[RET:.*]] = spirv.Constant dense<[-1, 3, 0]>
  %0 = spirv.ShiftRightArithmetic %c1, %c2 : vector<3xi32>, vector<3xi32>

  // CHECK: return %[[RET]]
  return %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.RightShiftLogical
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @lsr_x_0
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @lsr_x_0(%arg0 : i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  %c0 = spirv.Constant 0 : i32
  %cv0 = spirv.Constant dense<0> : vector<3xi32>

  %0 = spirv.ShiftRightLogical %arg0, %c0 : i32, i32
  %1 = spirv.ShiftRightLogical %arg1, %cv0 : vector<3xi32>, vector<3xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @lsr_shift_overflow
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @lsr_shift_overflow(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[C32:.*]] = spirv.Constant 32
  // CHECK-DAG: %[[CV:.*]] = spirv.Constant dense<[6, 18, 128]>
  %c32 = spirv.Constant 32 : i32
  %cv = spirv.Constant dense<[6, 18, 128]> : vector<3xi32>

  // CHECK: %0 = spirv.ShiftRightLogical %[[ARG0]], %[[C32]]
  // CHECK: %1 = spirv.ShiftRightLogical %[[ARG1]], %[[CV]]
  %0 = spirv.ShiftRightLogical %arg0, %c32 : i32, i32
  %1 = spirv.ShiftRightLogical %arg1, %cv : vector<3xi32>, vector<3xi32>
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @const_fold_scalar_lsr
func.func @const_fold_scalar_lsr() -> i32 {
  %c1 = spirv.Constant -131072 : i32  // 0xfffe 0000
  %c2 = spirv.Constant 17 : i32

  // 0x0000 ffff << 17 -> 0x0000 7fff
  // CHECK: %[[RET:.*]] = spirv.Constant 32767
  %0 = spirv.ShiftRightLogical %c1, %c2 : i32, i32

  // CHECK: return %[[RET]]
  return %0 : i32
}

// CHECK-LABEL: @const_fold_vector_lsr
func.func @const_fold_vector_lsr() -> vector<3xi32> {
  %c1 = spirv.Constant dense<[-2147483648, -1, -127]> : vector<3xi32>
  %c2 = spirv.Constant dense<[31, 16, 13]> : vector<3xi32>

  // CHECK: %[[RET:.*]] = spirv.Constant dense<[1, 65535, 524287]>
  %0 = spirv.ShiftRightLogical %c1, %c2 : vector<3xi32>, vector<3xi32>

  // CHECK: return %[[RET]]
  return %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitwiseAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_and_x_x
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @bitwise_and_x_x(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  %0 = spirv.BitwiseAnd %arg0, %arg0 : i32
  %1 = spirv.BitwiseAnd %arg1, %arg1 : vector<3xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @bitwise_and_x_0
func.func @bitwise_and_x_0(%arg0 : i32, %arg1 : vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0 : i32
  // CHECK-DAG: %[[CV0:.*]] = spirv.Constant dense<0> : vector<3xi32>
  %c0 = spirv.Constant 0 : i32
  %cv0 = spirv.Constant dense<0> : vector<3xi32>

  %0 = spirv.BitwiseAnd %arg0, %c0 : i32
  %1 = spirv.BitwiseAnd %arg1, %cv0 : vector<3xi32>

  // CHECK: return %[[C0]], %[[CV0]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @bitwise_and_x_n1
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @bitwise_and_x_n1(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  %cn1 = spirv.Constant -1 : i32
  %cvn1 = spirv.Constant dense<-1> : vector<3xi32>
  %0 = spirv.BitwiseAnd %arg0, %cn1 : i32
  %1 = spirv.BitwiseAnd %arg1, %cvn1 : vector<3xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @const_fold_scalar_band
func.func @const_fold_scalar_band() -> i32 {
  %c1 = spirv.Constant -268464129 : i32   // 0xefff 8fff
  %c2 = spirv.Constant 268464128: i32     // 0x1000 7000

  // 0xefff 8fff | 0x1000 7000 = 0xffff ffff = -1
  // CHECK: %[[C0:.*]] = spirv.Constant 0
  %0 = spirv.BitwiseAnd %c1, %c2 : i32

  // CHECK: return %[[C0]]
  return %0 : i32
}

// CHECK-LABEL: @const_fold_vector_band
func.func @const_fold_vector_band() -> vector<3xi32> {
  %c1 = spirv.Constant dense<[42, -55, 127]> : vector<3xi32>
  %c2 = spirv.Constant dense<[-3, -15, 28]> : vector<3xi32>

  // CHECK: %[[CV:.*]] = spirv.Constant dense<[40, -63, 28]>
  %0 = spirv.BitwiseAnd %c1, %c2 : vector<3xi32>

  // CHECK: return %[[CV]]
  return %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitwiseOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_or_x_x
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @bitwise_or_x_x(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  %0 = spirv.BitwiseOr %arg0, %arg0 : i32
  %1 = spirv.BitwiseOr %arg1, %arg1 : vector<3xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @bitwise_or_x_0
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @bitwise_or_x_0(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  %c1 = spirv.Constant 0 : i32
  %cv1 = spirv.Constant dense<0> : vector<3xi32>
  %0 = spirv.BitwiseOr %arg0, %c1 : i32
  %1 = spirv.BitwiseOr %arg1, %cv1 : vector<3xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @bitwise_or_x_n1
func.func @bitwise_or_x_n1(%arg0 : i32, %arg1 : vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[CN1:.*]] = spirv.Constant -1 : i32
  // CHECK-DAG: %[[CVN1:.*]] = spirv.Constant dense<-1> : vector<3xi32>
  %cn1 = spirv.Constant -1 : i32
  %cvn1 = spirv.Constant dense<-1> : vector<3xi32>
  %0 = spirv.BitwiseOr %arg0, %cn1 : i32
  %1 = spirv.BitwiseOr %arg1, %cvn1 : vector<3xi32>

  // CHECK: return %[[CN1]], %[[CVN1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @const_fold_scalar_bor
func.func @const_fold_scalar_bor() -> i32 {
  %c1 = spirv.Constant -268464129 : i32   // 0xefff 8fff
  %c2 = spirv.Constant 268464128: i32     // 0x1000 7000

  // 0xefff 8fff | 0x1000 7000 = 0xffff ffff = -1
  // CHECK: %[[CN1:.*]] = spirv.Constant -1
  %0 = spirv.BitwiseOr %c1, %c2 : i32

  // CHECK: return %[[CN1]]
  return %0 : i32
}

// CHECK-LABEL: @const_fold_vector_bor
func.func @const_fold_vector_bor() -> vector<3xi32> {
  %c1 = spirv.Constant dense<[42, -55, 127]> : vector<3xi32>
  %c2 = spirv.Constant dense<[-3, -15, 28]> : vector<3xi32>

  // CHECK: %[[CV:.*]] = spirv.Constant dense<[-1, -7, 127]>
  %0 = spirv.BitwiseOr %c1, %c2 : vector<3xi32>

  // CHECK: return %[[CV]]
  return %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BitwiseXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_xor_x_0
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: vector<3xi32>)
func.func @bitwise_xor_x_0(%arg0: i32, %arg1: vector<3xi32>) -> (i32, vector<3xi32>) {
  %c0 = spirv.Constant 0 : i32
  %cv0 = spirv.Constant dense<0> : vector<3xi32>

  %0 = spirv.BitwiseXor %arg0, %c0 : i32
  %1 = spirv.BitwiseXor %arg1, %cv0 : vector<3xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @bitwise_xor_x_x
func.func @bitwise_xor_x_x(%arg0 : i32, %arg1 : vector<3xi32>) -> (i32, vector<3xi32>) {
  // CHECK-DAG: %[[C0:.*]] = spirv.Constant 0
  // CHECK-DAG: %[[CV0:.*]] = spirv.Constant dense<0>
  %0 = spirv.BitwiseXor %arg0, %arg0 : i32
  %1 = spirv.BitwiseXor %arg1, %arg1 : vector<3xi32>

  // CHECK: return %[[C0]], %[[CV0]]
  return %0, %1 : i32, vector<3xi32>
}

// CHECK-LABEL: @const_fold_scalar_bxor
func.func @const_fold_scalar_bxor() -> i32 {
  %c1 = spirv.Constant 4294967295 : i32  // 2^32 - 1: 0xffff ffff
  %c2 = spirv.Constant -2147483648 : i32 // -2^31   : 0x8000 0000

  // 0x8000 0000 ^ 0xffff fffe = 0xefff ffff
  // CHECK: %[[CBIG:.*]] = spirv.Constant 2147483647
  %0 = spirv.BitwiseXor %c1, %c2 : i32

  // CHECK: return %[[CBIG]]
  return %0 : i32
}

// CHECK-LABEL: @const_fold_vector_bxor
func.func @const_fold_vector_bxor() -> vector<3xi32> {
  %c1 = spirv.Constant dense<[42, -55, 127]> : vector<3xi32>
  %c2 = spirv.Constant dense<[-3, -15, 28]> : vector<3xi32>

  // CHECK: %[[CV:.*]] = spirv.Constant dense<[-41, 56, 99]>
  %0 = spirv.BitwiseXor %c1, %c2 : vector<3xi32>

  // CHECK: return %[[CV]]
  return %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.mlir.selection
//===----------------------------------------------------------------------===//

func.func @canonicalize_selection_op_scalar_type(%cond: i1) -> () {
  %0 = spirv.Constant 0: i32
  // CHECK-DAG: %[[TRUE_VALUE:.*]] = spirv.Constant 1 : i32
  %1 = spirv.Constant 1: i32
  // CHECK-DAG: %[[FALSE_VALUE:.*]] = spirv.Constant 2 : i32
  %2 = spirv.Constant 2: i32
  // CHECK: %[[DST_VAR:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<i32, Function>
  %3 = spirv.Variable init(%0) : !spirv.ptr<i32, Function>

  // CHECK: %[[SRC_VALUE:.*]] = spirv.Select {{%.*}}, %[[TRUE_VALUE]], %[[FALSE_VALUE]] : i1, i32
  // CHECK-NEXT: spirv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE]] ["Aligned", 4] : i32
  // CHECK-NEXT: spirv.Return
  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^else

  ^else:
    spirv.Store "Function" %3, %2 ["Aligned", 4]: i32
    spirv.Branch ^merge

  ^then:
    spirv.Store "Function" %3, %1 ["Aligned", 4]: i32
    spirv.Branch ^merge

  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}

// -----

func.func @canonicalize_selection_op_vector_type(%cond: i1) -> () {
  %0 = spirv.Constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK-DAG: %[[TRUE_VALUE:.*]] = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-DAG: %[[FALSE_VALUE:.*]] = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<vector<3xi32>, Function>
  %3 = spirv.Variable init(%0) : !spirv.ptr<vector<3xi32>, Function>

  // CHECK: %[[SRC_VALUE:.*]] = spirv.Select {{%.*}}, %[[TRUE_VALUE]], %[[FALSE_VALUE]] : i1, vector<3xi32>
  // CHECK-NEXT: spirv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE]] ["Aligned", 8] : vector<3xi32>
  // CHECK-NEXT: spirv.Return
  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^else

  ^then:
    spirv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spirv.Branch ^merge

  ^else:
    spirv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spirv.Branch ^merge

  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}

// -----

// CHECK-LABEL: cannot_canonicalize_selection_op_0

// Store to a different variables.
func.func @cannot_canonicalize_selection_op_0(%cond: i1) -> () {
  %0 = spirv.Constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_1:.*]] = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_0:.*]] = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  %2 = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR_0:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<vector<3xi32>, Function>
  %3 = spirv.Variable init(%0) : !spirv.ptr<vector<3xi32>, Function>
  // CHECK: %[[DST_VAR_1:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<vector<3xi32>, Function>
  %4 = spirv.Variable init(%0) : !spirv.ptr<vector<3xi32>, Function>

  // CHECK: spirv.mlir.selection {
  spirv.mlir.selection {
    // CHECK: spirv.BranchConditional
    // CHECK-SAME: ^bb1(%[[DST_VAR_0]], %[[SRC_VALUE_0]]
    // CHECK-SAME: ^bb1(%[[DST_VAR_1]], %[[SRC_VALUE_1]]
    spirv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: ^bb1(%[[ARG0:.*]]: !spirv.ptr<vector<3xi32>, Function>, %[[ARG1:.*]]: vector<3xi32>):
    // CHECK: spirv.Store "Function" %[[ARG0]], %[[ARG1]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spirv.Branch ^merge

  ^else:
    spirv.Store "Function" %4, %2 ["Aligned", 8] : vector<3xi32>
    spirv.Branch ^merge

  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}

// -----

// CHECK-LABEL: cannot_canonicalize_selection_op_1

// A conditional block consists of more than 2 operations.
func.func @cannot_canonicalize_selection_op_1(%cond: i1) -> () {
  %0 = spirv.Constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_0:.*]] = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_1:.*]] = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR_0:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<vector<3xi32>, Function>
  %3 = spirv.Variable init(%0) : !spirv.ptr<vector<3xi32>, Function>
  // CHECK: %[[DST_VAR_1:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<vector<3xi32>, Function>
  %4 = spirv.Variable init(%0) : !spirv.ptr<vector<3xi32>, Function>

  // CHECK: spirv.mlir.selection {
  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spirv.Store "Function" %[[DST_VAR_0]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %3, %1 ["Aligned", 8] : vector<3xi32>
    // CHECK: spirv.Store "Function" %[[DST_VAR_1]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %4, %1 ["Aligned", 8]:  vector<3xi32>
    spirv.Branch ^merge

  ^else:
    // CHECK: spirv.Store "Function" %[[DST_VAR_1]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %4, %2 ["Aligned", 8] : vector<3xi32>
    spirv.Branch ^merge

  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}

// -----

// CHECK-LABEL: cannot_canonicalize_selection_op_2

// A control-flow goes into `^then` block from `^else` block.
func.func @cannot_canonicalize_selection_op_2(%cond: i1) -> () {
  %0 = spirv.Constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_0:.*]] = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_1:.*]] = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<vector<3xi32>, Function>
  %3 = spirv.Variable init(%0) : !spirv.ptr<vector<3xi32>, Function>

  // CHECK: spirv.mlir.selection {
  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spirv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spirv.Branch ^merge

  ^else:
    // CHECK: spirv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spirv.Branch ^then

  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}

// -----

// CHECK-LABEL: cannot_canonicalize_selection_op_3

// `spirv.Return` as a block terminator.
func.func @cannot_canonicalize_selection_op_3(%cond: i1) -> () {
  %0 = spirv.Constant dense<[0, 1, 2]> : vector<3xi32>
  %1 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_0:.*]] = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_1:.*]] = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<vector<3xi32>, Function>
  %3 = spirv.Variable init(%0) : !spirv.ptr<vector<3xi32>, Function>

  // CHECK: spirv.mlir.selection {
  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spirv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spirv.Return

  ^else:
    // CHECK: spirv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spirv.Branch ^merge

  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}

// -----

// CHECK-LABEL: cannot_canonicalize_selection_op_4

// Different memory access attributes.
func.func @cannot_canonicalize_selection_op_4(%cond: i1) -> () {
  %0 = spirv.Constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_0:.*]] = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spirv.Constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-DAG: %[[SRC_VALUE_1:.*]] = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spirv.Constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spirv.Variable init({{%.*}}) : !spirv.ptr<vector<3xi32>, Function>
  %3 = spirv.Variable init(%0) : !spirv.ptr<vector<3xi32>, Function>

  // CHECK: spirv.mlir.selection {
  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spirv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 4] : vector<3xi32>
    spirv.Store "Function" %3, %1 ["Aligned", 4]:  vector<3xi32>
    spirv.Branch ^merge

  ^else:
    // CHECK: spirv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spirv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spirv.Branch ^merge

  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}
