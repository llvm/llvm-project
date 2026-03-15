// RUN: mlir-opt -split-input-file -spirv-unify-aliased-resource -verify-diagnostics %s | FileCheck %s

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01v bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spirv.func @load_store_scalar(%index: i32) -> f32 "None" {
    %c0 = spirv.Constant 0 : i32
    %addr = spirv.mlir.addressof @var01s : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %value = spirv.Load "StorageBuffer" %ac : f32
    spirv.Store "StorageBuffer" %ac, %value : f32
    spirv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01s
//     CHECK: spirv.GlobalVariable @var01v bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01s

//     CHECK: spirv.func @load_store_scalar(%[[INDEX:.+]]: i32)
// CHECK-DAG:   %[[C0:.+]] = spirv.Constant 0 : i32
// CHECK-DAG:   %[[C4:.+]] = spirv.Constant 4 : i32
// CHECK-DAG:   %[[ADDR:.+]] = spirv.mlir.addressof @var01v
//     CHECK:   %[[DIV:.+]] = spirv.SDiv %[[INDEX]], %[[C4]] : i32
//     CHECK:   %[[MOD:.+]] = spirv.SMod %[[INDEX]], %[[C4]] : i32
//     CHECK:   %[[AC:.+]] = spirv.AccessChain %[[ADDR]][%[[C0]], %[[DIV]], %[[MOD]]]
//     CHECK:   spirv.Load "StorageBuffer" %[[AC]]
//     CHECK:   spirv.Store "StorageBuffer" %[[AC]]

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01v bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spirv.func @load_store_scalar_64bit(%index: i64) -> f32 "None" {
    %c0 = spirv.Constant 0 : i64
    %addr = spirv.mlir.addressof @var01s : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i64, i64 -> !spirv.ptr<f32, StorageBuffer>
    %value = spirv.Load "StorageBuffer" %ac : f32
    spirv.Store "StorageBuffer" %ac, %value : f32
    spirv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01s
//     CHECK: spirv.GlobalVariable @var01v bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01s

//     CHECK: spirv.func @load_store_scalar_64bit(%[[INDEX:.+]]: i64)
// CHECK-DAG:   %[[C4:.+]] = spirv.Constant 4 : i64
//     CHECK:   spirv.SDiv %[[INDEX]], %[[C4]] : i64
//     CHECK:   spirv.SMod %[[INDEX]], %[[C4]] : i64

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01v bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spirv.func @multiple_uses(%i0: i32, %i1: i32) -> f32 "None" {
    %c0 = spirv.Constant 0 : i32
    %addr = spirv.mlir.addressof @var01s : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac0 = spirv.AccessChain %addr[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %val0 = spirv.Load "StorageBuffer" %ac0 : f32
    %ac1 = spirv.AccessChain %addr[%c0, %i1] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %val1 = spirv.Load "StorageBuffer" %ac1 : f32
    %value = spirv.FAdd %val0, %val1 : f32
    spirv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01s
//     CHECK: spirv.GlobalVariable @var01v bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01s

//     CHECK: spirv.func @multiple_uses
//     CHECK:   %[[ADDR:.+]] = spirv.mlir.addressof @var01v
//     CHECK:   spirv.AccessChain %[[ADDR]][%{{.+}}, %{{.+}}, %{{.+}}]
//     CHECK:   spirv.AccessChain %[[ADDR]][%{{.+}}, %{{.+}}, %{{.+}}]

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01v bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<3xf32>, stride=16> [0])>, StorageBuffer>

  spirv.func @vector3(%index: i32) -> f32 "None" {
    %c0 = spirv.Constant 0 : i32
    %addr = spirv.mlir.addressof @var01s : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %value = spirv.Load "StorageBuffer" %ac : f32
    spirv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spirv.module

// CHECK: spirv.GlobalVariable @var01s bind(0, 1) {aliased}
// CHECK: spirv.GlobalVariable @var01v bind(0, 1) {aliased}
// CHECK: spirv.func @vector3

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01v bind(1, 0) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spirv.func @not_aliased(%index: i32) -> f32 "None" {
    %c0 = spirv.Constant 0 : i32
    %addr = spirv.mlir.addressof @var01s : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %value = spirv.Load "StorageBuffer" %ac : f32
    spirv.Store "StorageBuffer" %ac, %value : f32
    spirv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spirv.module

// CHECK: spirv.GlobalVariable @var01s bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
// CHECK: spirv.GlobalVariable @var01v bind(1, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK: spirv.func @not_aliased

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01s_1 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01v bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01v_1 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spirv.func @multiple_aliases(%index: i32) -> f32 "None" {
    %c0 = spirv.Constant 0 : i32

    %addr0 = spirv.mlir.addressof @var01s : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac0 = spirv.AccessChain %addr0[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %val0 = spirv.Load "StorageBuffer" %ac0 : f32

    %addr1 = spirv.mlir.addressof @var01s_1 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac1 = spirv.AccessChain %addr1[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %val1 = spirv.Load "StorageBuffer" %ac1 : f32

    %addr2 = spirv.mlir.addressof @var01v_1 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %ac2 = spirv.AccessChain %addr2[%c0, %index, %c0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %val2 = spirv.Load "StorageBuffer" %ac2 : f32

    %add0 = spirv.FAdd %val0, %val1 : f32
    %add1 = spirv.FAdd %add0, %val2 : f32
    spirv.ReturnValue %add1 : f32
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01s
//     CHECK: spirv.GlobalVariable @var01v bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01v_1

//     CHECK: spirv.func @multiple_aliases
//     CHECK:   %[[ADDR0:.+]] = spirv.mlir.addressof @var01v :
//     CHECK:   spirv.AccessChain %[[ADDR0]][%{{.+}}, %{{.+}}, %{{.+}}]
//     CHECK:   %[[ADDR1:.+]] = spirv.mlir.addressof @var01v :
//     CHECK:   spirv.AccessChain %[[ADDR1]][%{{.+}}, %{{.+}}, %{{.+}}]
//     CHECK:   %[[ADDR2:.+]] = spirv.mlir.addressof @var01v :
//     CHECK:   spirv.AccessChain %[[ADDR2]][%{{.+}}, %{{.+}}, %{{.+}}]

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s_i32 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01s_f32 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>

  spirv.func @different_scalar_type(%index: i32, %val1: f32) -> i32 "None" {
    %c0 = spirv.Constant 0 : i32

    %addr0 = spirv.mlir.addressof @var01s_i32 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %ac0 = spirv.AccessChain %addr0[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
    %val0 = spirv.Load "StorageBuffer" %ac0 : i32

    %addr1 = spirv.mlir.addressof @var01s_f32 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac1 = spirv.AccessChain %addr1[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    spirv.Store "StorageBuffer" %ac1, %val1 : f32

    spirv.ReturnValue %val0 : i32
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01s_f32
//     CHECK: spirv.GlobalVariable @var01s_i32 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// CHECK-NOT: @var01s_f32

//     CHECK: spirv.func @different_scalar_type(%[[INDEX:.+]]: i32, %[[VAL1:.+]]: f32)

//     CHECK:   %[[IADDR:.+]] = spirv.mlir.addressof @var01s_i32
//     CHECK:   %[[IAC:.+]] = spirv.AccessChain %[[IADDR]][%{{.+}}, %[[INDEX]]]
//     CHECK:   spirv.Load "StorageBuffer" %[[IAC]] : i32

//     CHECK:   %[[FADDR:.+]] = spirv.mlir.addressof @var01s_i32
//     CHECK:   %[[FAC:.+]] = spirv.AccessChain %[[FADDR]][%cst0_i32, %[[INDEX]]]
//     CHECK:   %[[CAST:.+]] = spirv.Bitcast %[[VAL1]] : f32 to i32
//     CHECK:   spirv.Store "StorageBuffer" %[[FAC]], %[[CAST]] : i32

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01v bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spirv.func @different_primitive_type(%index: i32, %val0: i32) -> i32 "None" {
    %c0 = spirv.Constant 0 : i32
    %addr = spirv.mlir.addressof @var01s : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i32, StorageBuffer>
    %val1 = spirv.Load "StorageBuffer" %ac : i32
    spirv.Store "StorageBuffer" %ac, %val0 : i32
    spirv.ReturnValue %val1 : i32
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01s
//     CHECK: spirv.GlobalVariable @var01v bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01s

//     CHECK: spirv.func @different_primitive_type(%{{.+}}: i32, %[[VAL0:.+]]: i32)
//     CHECK:   %[[ADDR:.+]] = spirv.mlir.addressof @var01v
//     CHECK:   %[[AC:.+]] = spirv.AccessChain %[[ADDR]][%{{.+}}, %{{.+}}, %{{.+}}]
//     CHECK:   %[[VAL1:.+]] = spirv.Load "StorageBuffer" %[[AC]] : f32
//     CHECK:   %[[CAST1:.+]] = spirv.Bitcast %[[VAL1]] : f32 to i32
//     CHECK:   %[[CAST2:.+]] = spirv.Bitcast %[[VAL0]] : i32 to f32
//     CHECK:   spirv.Store "StorageBuffer" %[[AC]], %[[CAST2]] : f32
//     CHECK:   spirv.ReturnValue %[[CAST1]] : i32

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s_i64 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01s_f32 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>

  spirv.func @load_different_scalar_bitwidth(%index: i32) -> i64 "None" {
    %c0 = spirv.Constant 0 : i32

    %addr0 = spirv.mlir.addressof @var01s_i64 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=4> [0])>, StorageBuffer>
    %ac0 = spirv.AccessChain %addr0[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i64, StorageBuffer>
    %val0 = spirv.Load "StorageBuffer" %ac0 : i64

    spirv.ReturnValue %val0 : i64
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01s_i64
//     CHECK: spirv.GlobalVariable @var01s_f32 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
// CHECK-NOT: @var01s_i64

//     CHECK: spirv.func @load_different_scalar_bitwidth(%[[INDEX:.+]]: i32)
//     CHECK:   %[[ZERO:.+]] = spirv.Constant 0 : i32
//     CHECK:   %[[ADDR:.+]] = spirv.mlir.addressof @var01s_f32

//     CHECK:   %[[TWO:.+]] = spirv.Constant 2 : i32
//     CHECK:   %[[BASE:.+]] = spirv.IMul %[[INDEX]], %[[TWO]] : i32
//     CHECK:   %[[AC0:.+]] = spirv.AccessChain %[[ADDR]][%[[ZERO]], %[[BASE]]]
//     CHECK:   %[[LOAD0:.+]] = spirv.Load "StorageBuffer" %[[AC0]] : f32

//     CHECK:   %[[ONE:.+]] = spirv.Constant 1 : i32
//     CHECK:   %[[ADD:.+]] = spirv.IAdd %[[BASE]], %[[ONE]] : i32
//     CHECK:   %[[AC1:.+]] = spirv.AccessChain %[[ADDR]][%[[ZERO]], %[[ADD]]]
//     CHECK:   %[[LOAD1:.+]] = spirv.Load "StorageBuffer" %[[AC1]] : f32

//     CHECK:   %[[CC:.+]] = spirv.CompositeConstruct %[[LOAD0]], %[[LOAD1]]
//     CHECK:   %[[CAST:.+]] = spirv.Bitcast %[[CC]] : vector<2xf32> to i64
//     CHECK:   spirv.ReturnValue %[[CAST]]

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01s_i64 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01s_f32 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>

  spirv.func @store_different_scalar_bitwidth(%i0: i32, %i1: i32) "None" {
    %c0 = spirv.Constant 0 : i32

    %addr0 = spirv.mlir.addressof @var01s_f32 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac0 = spirv.AccessChain %addr0[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %f32val = spirv.Load "StorageBuffer" %ac0 : f32
    %f64val = spirv.FConvert %f32val : f32 to f64
    %i64val = spirv.Bitcast %f64val : f64 to i64

    %addr1 = spirv.mlir.addressof @var01s_i64 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=4> [0])>, StorageBuffer>
    %ac1 = spirv.AccessChain %addr1[%c0, %i1] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i64, StorageBuffer>
    // expected-error@+1 {{failed to legalize operation 'spirv.Store'}}
    spirv.Store "StorageBuffer" %ac1, %i64val : i64

    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01_scalar bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01_vec2 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf32>, stride=8> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01_vec4 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spirv.func @load_different_vector_sizes(%i0: i32) -> vector<4xf32> "None" {
    %c0 = spirv.Constant 0 : i32

    %addr0 = spirv.mlir.addressof @var01_vec4 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %ac0 = spirv.AccessChain %addr0[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<vector<4xf32>, StorageBuffer>
    %vec4val = spirv.Load "StorageBuffer" %ac0 : vector<4xf32>

    %addr1 = spirv.mlir.addressof @var01_scalar : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac1 = spirv.AccessChain %addr1[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %scalarval = spirv.Load "StorageBuffer" %ac1 : f32

    %val = spirv.CompositeInsert %scalarval, %vec4val[0 : i32] : f32 into vector<4xf32>
    spirv.ReturnValue %val : vector<4xf32>
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01_scalar
// CHECK-NOT: @var01_vec4
//     CHECK: spirv.GlobalVariable @var01_vec2 bind(0, 1) : !spirv.ptr<{{.+}}>
// CHECK-NOT: @var01_scalar
// CHECK-NOT: @var01_vec4

//     CHECK: spirv.func @load_different_vector_sizes(%[[IDX:.+]]: i32)
//     CHECK:   %[[ZERO:.+]] = spirv.Constant 0 : i32
//     CHECK:   %[[ADDR:.+]] = spirv.mlir.addressof @var01_vec2
//     CHECK:   %[[TWO:.+]] = spirv.Constant 2 : i32
//     CHECK:   %[[IDX0:.+]] = spirv.IMul %[[IDX]], %[[TWO]] : i32
//     CHECK:   %[[AC0:.+]] = spirv.AccessChain %[[ADDR]][%[[ZERO]], %[[IDX0]]]
//     CHECK:   %[[LD0:.+]] = spirv.Load "StorageBuffer" %[[AC0]] : vector<2xf32>
//     CHECK:   %[[ONE:.+]] = spirv.Constant 1 : i32
//     CHECK:   %[[IDX1:.+]] = spirv.IAdd %0, %[[ONE]] : i32
//     CHECK:   %[[AC1:.+]] = spirv.AccessChain %[[ADDR]][%[[ZERO]], %[[IDX1]]]
//     CHECK:   %[[LD1:.+]] = spirv.Load "StorageBuffer" %[[AC1]] : vector<2xf32>
//     CHECK:   spirv.CompositeConstruct %[[LD0]], %[[LD1]] : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>

//     CHECK:   %[[ADDR:.+]] = spirv.mlir.addressof @var01_vec2
//     CHECK:   %[[TWO:.+]] = spirv.Constant 2 : i32
//     CHECK:   %[[DIV:.+]] = spirv.SDiv %[[IDX]], %[[TWO]] : i32
//     CHECK:   %[[MOD:.+]] = spirv.SMod %[[IDX]], %[[TWO]] : i32
//     CHECK:   %[[AC:.+]] = spirv.AccessChain %[[ADDR]][%[[ZERO]], %[[DIV]], %[[MOD]]]
//     CHECK:   %[[LD:.+]] = spirv.Load "StorageBuffer" %[[AC]] : f32

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01_v4f32 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01_f32 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01_i64 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=8> [0])>, StorageBuffer>

  spirv.func @load_mixed_scalar_vector_primitive_types(%i0: i32) -> vector<4xf32> "None" {
    %c0 = spirv.Constant 0 : i32

    %addr0 = spirv.mlir.addressof @var01_v4f32 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %ac0 = spirv.AccessChain %addr0[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<vector<4xf32>, StorageBuffer>
    %vec4val = spirv.Load "StorageBuffer" %ac0 : vector<4xf32>

    %addr1 = spirv.mlir.addressof @var01_f32 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac1 = spirv.AccessChain %addr1[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %f32val = spirv.Load "StorageBuffer" %ac1 : f32

    %addr2 = spirv.mlir.addressof @var01_i64 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=8> [0])>, StorageBuffer>
    %ac2 = spirv.AccessChain %addr2[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=8> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i64, StorageBuffer>
    %i64val = spirv.Load "StorageBuffer" %ac2 : i64
    %i32val = spirv.SConvert %i64val : i64 to i32
    %castval = spirv.Bitcast %i32val : i32 to f32

    %val1 = spirv.CompositeInsert %f32val, %vec4val[0 : i32] : f32 into vector<4xf32>
    %val2 = spirv.CompositeInsert %castval, %val1[1 : i32] : f32 into vector<4xf32>
    spirv.ReturnValue %val2 : vector<4xf32>
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var01_f32
// CHECK-NOT: @var01_i64
//     CHECK: spirv.GlobalVariable @var01_v4f32 bind(0, 1) : !spirv.ptr<{{.+}}>
// CHECK-NOT: @var01_f32
// CHECK-NOT: @var01_i64

// CHECK:  spirv.func @load_mixed_scalar_vector_primitive_types(%[[IDX:.+]]: i32)

// CHECK:    %[[ZERO:.+]] = spirv.Constant 0 : i32
// CHECK:    %[[ADDR0:.+]] = spirv.mlir.addressof @var01_v4f32
// CHECK:    %[[AC0:.+]] = spirv.AccessChain %[[ADDR0]][%[[ZERO]], %[[IDX]]]
// CHECK:    spirv.Load "StorageBuffer" %[[AC0]] : vector<4xf32>

// CHECK:    %[[ADDR1:.+]] = spirv.mlir.addressof @var01_v4f32
// CHECK:    %[[FOUR:.+]] = spirv.Constant 4 : i32
// CHECK:    %[[DIV:.+]] = spirv.SDiv %[[IDX]], %[[FOUR]] : i32
// CHECK:    %[[MOD:.+]] = spirv.SMod %[[IDX]], %[[FOUR]] : i32
// CHECK:    %[[AC1:.+]] = spirv.AccessChain %[[ADDR1]][%[[ZERO]], %[[DIV]], %[[MOD]]]
// CHECK:    spirv.Load "StorageBuffer" %[[AC1]] : f32

// CHECK:    %[[ADDR2:.+]] = spirv.mlir.addressof @var01_v4f32
// CHECK:    %[[TWO:.+]] = spirv.Constant 2 : i32
// CHECK:    %[[DIV0:.+]] = spirv.SDiv %[[IDX]], %[[TWO]] : i32
// CHECK:    %[[MOD0:.+]] = spirv.SMod %[[IDX]], %[[TWO]] : i32
// CHECK:    %[[AC2:.+]] = spirv.AccessChain %[[ADDR2]][%[[ZERO]], %[[DIV0]], %[[MOD0]]]
// CHECK:    %[[LD0:.+]] = spirv.Load "StorageBuffer" %[[AC2]] : f32

// CHECK:    %[[ONE:.+]] = spirv.Constant 1 : i32
// CHECK:    %[[MOD1:.+]] = spirv.IAdd %[[MOD0]], %[[ONE]]
// CHECK:    %[[AC3:.+]] = spirv.AccessChain %[[ADDR2]][%[[ZERO]], %[[DIV0]], %[[MOD1]]]
// CHECK:    %[[LD1:.+]] = spirv.Load "StorageBuffer" %[[AC3]] : f32
// CHECK:    %[[CC:.+]] = spirv.CompositeConstruct %[[LD0]], %[[LD1]]
// CHECK:    %[[BC:.+]] = spirv.Bitcast %[[CC]] : vector<2xf32> to i64

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01_v2f2 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf32>, stride=16> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01_i64 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=8> [0])>, StorageBuffer>

  spirv.func @load_mixed_scalar_vector_primitive_types(%i0: i32) -> i64 "None" {
    %c0 = spirv.Constant 0 : i32

    %addr = spirv.mlir.addressof @var01_i64 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=8> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i64, stride=8> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i64, StorageBuffer>
    %val = spirv.Load "StorageBuffer" %ac : i64

    spirv.ReturnValue %val : i64
  }
}

// CHECK-LABEL: spirv.module

// CHECK:  spirv.func @load_mixed_scalar_vector_primitive_types(%[[IDX:.+]]: i32)

// CHECK:    %[[ADDR:.+]] = spirv.mlir.addressof @var01_v2f2
// CHECK:    %[[ONE:.+]] = spirv.Constant 1 : i32
// CHECK:    %[[DIV:.+]] = spirv.SDiv %[[IDX]], %[[ONE]] : i32
// CHECK:    %[[MOD:.+]] = spirv.SMod %[[IDX]], %[[ONE]] : i32
// CHECK:    spirv.AccessChain %[[ADDR]][%{{.+}}, %[[DIV]], %[[MOD]]]
// CHECK:    spirv.Load
// CHECK:    spirv.Load

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01_v2f2 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf32>, stride=16> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01_i16 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i16, stride=2> [0])>, StorageBuffer>

  spirv.func @scalar_type_bitwidth_smaller_than_vector(%i0: i32) -> i16 "None" {
    %c0 = spirv.Constant 0 : i32

    %addr = spirv.mlir.addressof @var01_i16 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i16, stride=2> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<i16, stride=2> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<i16, StorageBuffer>
    %val = spirv.Load "StorageBuffer" %ac : i16

    spirv.ReturnValue %val : i16
  }
}

// CHECK-LABEL: spirv.module

// CHECK: spirv.GlobalVariable @var01_v2f2 bind(0, 1) {aliased}
// CHECK: spirv.GlobalVariable @var01_i16 bind(0, 1) {aliased}

// CHECK: spirv.func @scalar_type_bitwidth_smaller_than_vector

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var00_v4f32 bind(0, 0) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spirv.GlobalVariable @var00_v4f16 bind(0, 0) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf16>, stride=8> [0])>, StorageBuffer>

  spirv.func @vector_type_same_size_different_element_type(%i0: i32) -> vector<4xf32> "None" {
    %c0 = spirv.Constant 0 : i32

    %addr = spirv.mlir.addressof @var00_v4f32 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<vector<4xf32>, StorageBuffer>
    %val = spirv.Load "StorageBuffer" %ac :  vector<4xf32>

    spirv.ReturnValue %val : vector<4xf32>
  }
}

// CHECK-LABEL: spirv.module

// CHECK: spirv.GlobalVariable @var00_v4f16 bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf16>, stride=8> [0])>, StorageBuffer>

// CHECK: spirv.func @vector_type_same_size_different_element_type

// CHECK:   %[[LD0:.+]] = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf16>
// CHECK:   %[[LD1:.+]] = spirv.Load "StorageBuffer" %{{.+}} : vector<4xf16>
// CHECK:   %[[BC0:.+]] = spirv.Bitcast %[[LD0]] : vector<4xf16> to vector<2xf32>
// CHECK:   %[[BC1:.+]] = spirv.Bitcast %[[LD1]] : vector<4xf16> to vector<2xf32>
// CHECK:   %[[CC:.+]] = spirv.CompositeConstruct %[[BC0]], %[[BC1]] : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
// CHECK:   spirv.ReturnValue %[[CC]]

// -----

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var01_v2f16 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf16>, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @var01_v2f32 bind(0, 1) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf32>, stride=8> [0])>, StorageBuffer>

  spirv.func @aliased(%index: i32) -> vector<3xf32> "None" {
    %c0 = spirv.Constant 0 : i32
    %v0 = spirv.Constant dense<0.0> : vector<3xf32>
    %addr0 = spirv.mlir.addressof @var01_v2f16 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf16>, stride=4> [0])>, StorageBuffer>
    %ac0 = spirv.AccessChain %addr0[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf16>, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<vector<2xf16>, StorageBuffer>
    %value0 = spirv.Load "StorageBuffer" %ac0 : vector<2xf16>

    %addr1 = spirv.mlir.addressof @var01_v2f32 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf32>, stride=8> [0])>, StorageBuffer>
    %ac1 = spirv.AccessChain %addr1[%c0, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf32>, stride=8> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<vector<2xf32>, StorageBuffer>
    %value1 = spirv.Load "StorageBuffer" %ac1 : vector<2xf32>

    %val0_as_f32 = spirv.Bitcast %value0 : vector<2xf16> to f32

    %res = spirv.CompositeConstruct %val0_as_f32, %value1 : (f32, vector<2xf32>) -> vector<3xf32>

    spirv.ReturnValue %res : vector<3xf32>
  }
}

// CHECK-LABEL: spirv.module

// CHECK: spirv.GlobalVariable @var01_v2f16 bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf16>, stride=4> [0])>, StorageBuffer>
// CHECK: spirv.func @aliased

// CHECK:     %[[LD0:.+]] = spirv.Load "StorageBuffer" %{{.+}} : vector<2xf16>
// CHECK:     %[[LD1:.+]] = spirv.Load "StorageBuffer" %{{.+}} : vector<2xf16>
// CHECK:     %[[LD2:.+]] = spirv.Load "StorageBuffer" %{{.+}} : vector<2xf16>

// CHECK-DAG: %[[ELEM0:.+]] = spirv.Bitcast %[[LD0]] : vector<2xf16> to f32
// CHECK-DAG: %[[ELEM1:.+]] = spirv.Bitcast %[[LD1]] : vector<2xf16> to f32
// CHECK-DAG: %[[ELEM2:.+]] = spirv.Bitcast %[[LD2]] : vector<2xf16> to f32

// CHECK:     %[[RES:.+]] = spirv.CompositeConstruct %[[ELEM0]], %{{.+}} : (f32, vector<2xf32>) -> vector<3xf32>
// CHECK:     spirv.ReturnValue %[[RES]] : vector<3xf32>

// -----

// Make sure we do not crash on function arguments.

spirv.module Logical GLSL450 {
  spirv.func @main(%arg0: !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>) "None" {
    %cst0_i32 = spirv.Constant 0 : i32
    %0 = spirv.AccessChain %arg0[%cst0_i32, %cst0_i32] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    spirv.Return
  }
}

// CHECK-LABEL: spirv.module
// CHECK-LABEL: spirv.func @main
// CHECK-SAME:  (%{{.+}}: !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>) "None"
// CHECK:       spirv.Return
