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
// spirv.UMod
//===----------------------------------------------------------------------===//

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
