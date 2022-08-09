// RUN: mlir-opt -split-input-file -spirv-unify-aliased-resource -verify-diagnostics %s | FileCheck %s

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01v bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spv.func @load_store_scalar(%index: i32) -> f32 "None" {
    %c0 = spv.Constant 0 : i32
    %addr = spv.mlir.addressof @var01s : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac = spv.AccessChain %addr[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %value = spv.Load "StorageBuffer" %ac : f32
    spv.Store "StorageBuffer" %ac, %value : f32
    spv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spv.module

// CHECK-NOT: @var01s
//     CHECK: spv.GlobalVariable @var01v bind(0, 1) : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01s

//     CHECK: spv.func @load_store_scalar(%[[INDEX:.+]]: i32)
// CHECK-DAG:   %[[C0:.+]] = spv.Constant 0 : i32
// CHECK-DAG:   %[[C4:.+]] = spv.Constant 4 : i32
// CHECK-DAG:   %[[ADDR:.+]] = spv.mlir.addressof @var01v
//     CHECK:   %[[DIV:.+]] = spv.SDiv %[[INDEX]], %[[C4]] : i32
//     CHECK:   %[[MOD:.+]] = spv.SMod %[[INDEX]], %[[C4]] : i32
//     CHECK:   %[[AC:.+]] = spv.AccessChain %[[ADDR]][%[[C0]], %[[DIV]], %[[MOD]]]
//     CHECK:   spv.Load "StorageBuffer" %[[AC]]
//     CHECK:   spv.Store "StorageBuffer" %[[AC]]

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01v bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spv.func @multiple_uses(%i0: i32, %i1: i32) -> f32 "None" {
    %c0 = spv.Constant 0 : i32
    %addr = spv.mlir.addressof @var01s : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac0 = spv.AccessChain %addr[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %val0 = spv.Load "StorageBuffer" %ac0 : f32
    %ac1 = spv.AccessChain %addr[%c0, %i1] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %val1 = spv.Load "StorageBuffer" %ac1 : f32
    %value = spv.FAdd %val0, %val1 : f32
    spv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spv.module

// CHECK-NOT: @var01s
//     CHECK: spv.GlobalVariable @var01v bind(0, 1) : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01s

//     CHECK: spv.func @multiple_uses
//     CHECK:   %[[ADDR:.+]] = spv.mlir.addressof @var01v
//     CHECK:   spv.AccessChain %[[ADDR]][%{{.+}}, %{{.+}}, %{{.+}}]
//     CHECK:   spv.AccessChain %[[ADDR]][%{{.+}}, %{{.+}}, %{{.+}}]

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01v bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<3xf32>, stride=16> [0])>, StorageBuffer>

  spv.func @vector3(%index: i32) -> f32 "None" {
    %c0 = spv.Constant 0 : i32
    %addr = spv.mlir.addressof @var01s : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac = spv.AccessChain %addr[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %value = spv.Load "StorageBuffer" %ac : f32
    spv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spv.module

// CHECK: spv.GlobalVariable @var01s bind(0, 1) {aliased}
// CHECK: spv.GlobalVariable @var01v bind(0, 1) {aliased}
// CHECK: spv.func @vector3

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01v bind(1, 0) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spv.func @not_aliased(%index: i32) -> f32 "None" {
    %c0 = spv.Constant 0 : i32
    %addr = spv.mlir.addressof @var01s : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac = spv.AccessChain %addr[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %value = spv.Load "StorageBuffer" %ac : f32
    spv.Store "StorageBuffer" %ac, %value : f32
    spv.ReturnValue %value : f32
  }
}

// CHECK-LABEL: spv.module

// CHECK: spv.GlobalVariable @var01s bind(0, 1) : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
// CHECK: spv.GlobalVariable @var01v bind(1, 0) : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK: spv.func @not_aliased

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01s_1 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01v bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.GlobalVariable @var01v_1 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spv.func @multiple_aliases(%index: i32) -> f32 "None" {
    %c0 = spv.Constant 0 : i32

    %addr0 = spv.mlir.addressof @var01s : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac0 = spv.AccessChain %addr0[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %val0 = spv.Load "StorageBuffer" %ac0 : f32

    %addr1 = spv.mlir.addressof @var01s_1 : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac1 = spv.AccessChain %addr1[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %val1 = spv.Load "StorageBuffer" %ac1 : f32

    %addr2 = spv.mlir.addressof @var01v_1 : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %ac2 = spv.AccessChain %addr2[%c0, %index, %c0] : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32, i32
    %val2 = spv.Load "StorageBuffer" %ac2 : f32

    %add0 = spv.FAdd %val0, %val1 : f32
    %add1 = spv.FAdd %add0, %val2 : f32
    spv.ReturnValue %add1 : f32
  }
}

// CHECK-LABEL: spv.module

// CHECK-NOT: @var01s
//     CHECK: spv.GlobalVariable @var01v bind(0, 1) : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01v_1

//     CHECK: spv.func @multiple_aliases
//     CHECK:   %[[ADDR0:.+]] = spv.mlir.addressof @var01v :
//     CHECK:   spv.AccessChain %[[ADDR0]][%{{.+}}, %{{.+}}, %{{.+}}]
//     CHECK:   %[[ADDR1:.+]] = spv.mlir.addressof @var01v :
//     CHECK:   spv.AccessChain %[[ADDR1]][%{{.+}}, %{{.+}}, %{{.+}}]
//     CHECK:   %[[ADDR2:.+]] = spv.mlir.addressof @var01v :
//     CHECK:   spv.AccessChain %[[ADDR2]][%{{.+}}, %{{.+}}, %{{.+}}]

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s_i32 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01s_f32 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>

  spv.func @different_scalar_type(%index: i32, %val1: f32) -> i32 "None" {
    %c0 = spv.Constant 0 : i32

    %addr0 = spv.mlir.addressof @var01s_i32 : !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %ac0 = spv.AccessChain %addr0[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32
    %val0 = spv.Load "StorageBuffer" %ac0 : i32

    %addr1 = spv.mlir.addressof @var01s_f32 : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac1 = spv.AccessChain %addr1[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    spv.Store "StorageBuffer" %ac1, %val1 : f32

    spv.ReturnValue %val0 : i32
  }
}

// CHECK-LABEL: spv.module

// CHECK-NOT: @var01s_f32
//     CHECK: spv.GlobalVariable @var01s_i32 bind(0, 1) : !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// CHECK-NOT: @var01s_f32

//     CHECK: spv.func @different_scalar_type(%[[INDEX:.+]]: i32, %[[VAL1:.+]]: f32)

//     CHECK:   %[[IADDR:.+]] = spv.mlir.addressof @var01s_i32
//     CHECK:   %[[IAC:.+]] = spv.AccessChain %[[IADDR]][%{{.+}}, %[[INDEX]]]
//     CHECK:   spv.Load "StorageBuffer" %[[IAC]] : i32

//     CHECK:   %[[FADDR:.+]] = spv.mlir.addressof @var01s_i32
//     CHECK:   %[[FAC:.+]] = spv.AccessChain %[[FADDR]][%cst0_i32, %[[INDEX]]]
//     CHECK:   %[[CAST:.+]] = spv.Bitcast %[[VAL1]] : f32 to i32
//     CHECK:   spv.Store "StorageBuffer" %[[FAC]], %[[CAST]] : i32

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01v bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spv.func @different_primitive_type(%index: i32, %val0: i32) -> i32 "None" {
    %c0 = spv.Constant 0 : i32
    %addr = spv.mlir.addressof @var01s : !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
    %ac = spv.AccessChain %addr[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>, i32, i32
    %val1 = spv.Load "StorageBuffer" %ac : i32
    spv.Store "StorageBuffer" %ac, %val0 : i32
    spv.ReturnValue %val1 : i32
  }
}

// CHECK-LABEL: spv.module

// CHECK-NOT: @var01s
//     CHECK: spv.GlobalVariable @var01v bind(0, 1) : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
// CHECK-NOT: @var01s

//     CHECK: spv.func @different_primitive_type(%{{.+}}: i32, %[[VAL0:.+]]: i32)
//     CHECK:   %[[ADDR:.+]] = spv.mlir.addressof @var01v
//     CHECK:   %[[AC:.+]] = spv.AccessChain %[[ADDR]][%{{.+}}, %{{.+}}, %{{.+}}]
//     CHECK:   %[[VAL1:.+]] = spv.Load "StorageBuffer" %[[AC]] : f32
//     CHECK:   %[[CAST1:.+]] = spv.Bitcast %[[VAL1]] : f32 to i32
//     CHECK:   %[[CAST2:.+]] = spv.Bitcast %[[VAL0]] : i32 to f32
//     CHECK:   spv.Store "StorageBuffer" %[[AC]], %[[CAST2]] : f32
//     CHECK:   spv.ReturnValue %[[CAST1]] : i32

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s_i64 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01s_f32 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>

  spv.func @load_different_scalar_bitwidth(%index: i32) -> i64 "None" {
    %c0 = spv.Constant 0 : i32

    %addr0 = spv.mlir.addressof @var01s_i64 : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=4> [0])>, StorageBuffer>
    %ac0 = spv.AccessChain %addr0[%c0, %index] : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=4> [0])>, StorageBuffer>, i32, i32
    %val0 = spv.Load "StorageBuffer" %ac0 : i64

    spv.ReturnValue %val0 : i64
  }
}

// CHECK-LABEL: spv.module

// CHECK-NOT: @var01s_i64
//     CHECK: spv.GlobalVariable @var01s_f32 bind(0, 1) : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
// CHECK-NOT: @var01s_i64

//     CHECK: spv.func @load_different_scalar_bitwidth(%[[INDEX:.+]]: i32)
//     CHECK:   %[[ZERO:.+]] = spv.Constant 0 : i32
//     CHECK:   %[[ADDR:.+]] = spv.mlir.addressof @var01s_f32

//     CHECK:   %[[TWO:.+]] = spv.Constant 2 : i32
//     CHECK:   %[[BASE:.+]] = spv.IMul %[[INDEX]], %[[TWO]] : i32
//     CHECK:   %[[AC0:.+]] = spv.AccessChain %[[ADDR]][%[[ZERO]], %[[BASE]]]
//     CHECK:   %[[LOAD0:.+]] = spv.Load "StorageBuffer" %[[AC0]] : f32

//     CHECK:   %[[ONE:.+]] = spv.Constant 1 : i32
//     CHECK:   %[[ADD:.+]] = spv.IAdd %[[BASE]], %[[ONE]] : i32
//     CHECK:   %[[AC1:.+]] = spv.AccessChain %[[ADDR]][%[[ZERO]], %[[ADD]]]
//     CHECK:   %[[LOAD1:.+]] = spv.Load "StorageBuffer" %[[AC1]] : f32

//     CHECK:   %[[CC:.+]] = spv.CompositeConstruct %[[LOAD0]], %[[LOAD1]]
//     CHECK:   %[[CAST:.+]] = spv.Bitcast %[[CC]] : vector<2xf32> to i64
//     CHECK:   spv.ReturnValue %[[CAST]]

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01s_i64 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01s_f32 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>

  spv.func @store_different_scalar_bitwidth(%i0: i32, %i1: i32) "None" {
    %c0 = spv.Constant 0 : i32

    %addr0 = spv.mlir.addressof @var01s_f32 : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac0 = spv.AccessChain %addr0[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %f32val = spv.Load "StorageBuffer" %ac0 : f32
    %f64val = spv.FConvert %f32val : f32 to f64
    %i64val = spv.Bitcast %f64val : f64 to i64

    %addr1 = spv.mlir.addressof @var01s_i64 : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=4> [0])>, StorageBuffer>
    %ac1 = spv.AccessChain %addr1[%c0, %i1] : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=4> [0])>, StorageBuffer>, i32, i32
    // expected-error@+1 {{failed to legalize operation 'spv.Store'}}
    spv.Store "StorageBuffer" %ac1, %i64val : i64

    spv.Return
  }
}

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01_scalar bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01_vec2 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<2xf32>, stride=8> [0])>, StorageBuffer>
  spv.GlobalVariable @var01_vec4 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>

  spv.func @load_different_vector_sizes(%i0: i32) -> vector<4xf32> "None" {
    %c0 = spv.Constant 0 : i32

    %addr0 = spv.mlir.addressof @var01_vec4 : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %ac0 = spv.AccessChain %addr0[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %vec4val = spv.Load "StorageBuffer" %ac0 : vector<4xf32>

    %addr1 = spv.mlir.addressof @var01_scalar : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac1 = spv.AccessChain %addr1[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %scalarval = spv.Load "StorageBuffer" %ac1 : f32

    %val = spv.CompositeInsert %scalarval, %vec4val[0 : i32] : f32 into vector<4xf32>
    spv.ReturnValue %val : vector<4xf32>
  }
}

// CHECK-LABEL: spv.module

// CHECK-NOT: @var01_scalar
// CHECK-NOT: @var01_vec4
//     CHECK: spv.GlobalVariable @var01_vec2 bind(0, 1) : !spv.ptr<{{.+}}>
// CHECK-NOT: @var01_scalar
// CHECK-NOT: @var01_vec4

//     CHECK: spv.func @load_different_vector_sizes(%[[IDX:.+]]: i32)
//     CHECK:   %[[ZERO:.+]] = spv.Constant 0 : i32
//     CHECK:   %[[ADDR:.+]] = spv.mlir.addressof @var01_vec2
//     CHECK:   %[[TWO:.+]] = spv.Constant 2 : i32
//     CHECK:   %[[IDX0:.+]] = spv.IMul %[[IDX]], %[[TWO]] : i32
//     CHECK:   %[[AC0:.+]] = spv.AccessChain %[[ADDR]][%[[ZERO]], %[[IDX0]]]
//     CHECK:   %[[LD0:.+]] = spv.Load "StorageBuffer" %[[AC0]] : vector<2xf32>
//     CHECK:   %[[ONE:.+]] = spv.Constant 1 : i32
//     CHECK:   %[[IDX1:.+]] = spv.IAdd %0, %[[ONE]] : i32
//     CHECK:   %[[AC1:.+]] = spv.AccessChain %[[ADDR]][%[[ZERO]], %[[IDX1]]]
//     CHECK:   %[[LD1:.+]] = spv.Load "StorageBuffer" %[[AC1]] : vector<2xf32>
//     CHECK:   spv.CompositeConstruct %[[LD0]], %[[LD1]] : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>

//     CHECK:   %[[ADDR:.+]] = spv.mlir.addressof @var01_vec2
//     CHECK:   %[[TWO:.+]] = spv.Constant 2 : i32
//     CHECK:   %[[DIV:.+]] = spv.SDiv %[[IDX]], %[[TWO]] : i32
//     CHECK:   %[[MOD:.+]] = spv.SMod %[[IDX]], %[[TWO]] : i32
//     CHECK:   %[[AC:.+]] = spv.AccessChain %[[ADDR]][%[[ZERO]], %[[DIV]], %[[MOD]]]
//     CHECK:   %[[LD:.+]] = spv.Load "StorageBuffer" %[[AC]] : f32

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01_v4f32 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
  spv.GlobalVariable @var01_f32 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @var01_i64 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=8> [0])>, StorageBuffer>

  spv.func @load_mixed_scalar_vector_primitive_types(%i0: i32) -> vector<4xf32> "None" {
    %c0 = spv.Constant 0 : i32

    %addr0 = spv.mlir.addressof @var01_v4f32 : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
    %ac0 = spv.AccessChain %addr0[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32
    %vec4val = spv.Load "StorageBuffer" %ac0 : vector<4xf32>

    %addr1 = spv.mlir.addressof @var01_f32 : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac1 = spv.AccessChain %addr1[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %f32val = spv.Load "StorageBuffer" %ac1 : f32

    %addr2 = spv.mlir.addressof @var01_i64 : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=8> [0])>, StorageBuffer>
    %ac2 = spv.AccessChain %addr2[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=8> [0])>, StorageBuffer>, i32, i32
    %i64val = spv.Load "StorageBuffer" %ac2 : i64
    %i32val = spv.SConvert %i64val : i64 to i32
    %castval = spv.Bitcast %i32val : i32 to f32

    %val1 = spv.CompositeInsert %f32val, %vec4val[0 : i32] : f32 into vector<4xf32>
    %val2 = spv.CompositeInsert %castval, %val1[1 : i32] : f32 into vector<4xf32>
    spv.ReturnValue %val2 : vector<4xf32>
  }
}

// CHECK-LABEL: spv.module

// CHECK-NOT: @var01_f32
// CHECK-NOT: @var01_i64
//     CHECK: spv.GlobalVariable @var01_v4f32 bind(0, 1) : !spv.ptr<{{.+}}>
// CHECK-NOT: @var01_f32
// CHECK-NOT: @var01_i64

// CHECK:  spv.func @load_mixed_scalar_vector_primitive_types(%[[IDX:.+]]: i32)

// CHECK:    %[[ZERO:.+]] = spv.Constant 0 : i32
// CHECK:    %[[ADDR0:.+]] = spv.mlir.addressof @var01_v4f32
// CHECK:    %[[AC0:.+]] = spv.AccessChain %[[ADDR0]][%[[ZERO]], %[[IDX]]]
// CHECK:    spv.Load "StorageBuffer" %[[AC0]] : vector<4xf32>

// CHECK:    %[[ADDR1:.+]] = spv.mlir.addressof @var01_v4f32
// CHECK:    %[[FOUR:.+]] = spv.Constant 4 : i32
// CHECK:    %[[DIV:.+]] = spv.SDiv %[[IDX]], %[[FOUR]] : i32
// CHECK:    %[[MOD:.+]] = spv.SMod %[[IDX]], %[[FOUR]] : i32
// CHECK:    %[[AC1:.+]] = spv.AccessChain %[[ADDR1]][%[[ZERO]], %[[DIV]], %[[MOD]]]
// CHECK:    spv.Load "StorageBuffer" %[[AC1]] : f32

// CHECK:    %[[ADDR2:.+]] = spv.mlir.addressof @var01_v4f32
// CHECK:    %[[TWO:.+]] = spv.Constant 2 : i32
// CHECK:    %[[DIV0:.+]] = spv.SDiv %[[IDX]], %[[TWO]] : i32
// CHECK:    %[[MOD0:.+]] = spv.SMod %[[IDX]], %[[TWO]] : i32
// CHECK:    %[[AC2:.+]] = spv.AccessChain %[[ADDR2]][%[[ZERO]], %[[DIV0]], %[[MOD0]]]
// CHECK:    %[[LD0:.+]] = spv.Load "StorageBuffer" %[[AC2]] : f32

// CHECK:    %[[ONE:.+]] = spv.Constant 1 : i32
// CHECK:    %[[MOD1:.+]] = spv.IAdd %[[MOD0]], %[[ONE]]
// CHECK:    %[[AC3:.+]] = spv.AccessChain %[[ADDR2]][%[[ZERO]], %[[DIV0]], %[[MOD1]]]
// CHECK:    %[[LD1:.+]] = spv.Load "StorageBuffer" %[[AC3]] : f32
// CHECK:    %[[CC:.+]] = spv.CompositeConstruct %[[LD0]], %[[LD1]]
// CHECK:    %[[BC:.+]] = spv.Bitcast %[[CC]] : vector<2xf32> to i64

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01_v2f2 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<2xf32>, stride=16> [0])>, StorageBuffer>
  spv.GlobalVariable @var01_i64 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=8> [0])>, StorageBuffer>

  spv.func @load_mixed_scalar_vector_primitive_types(%i0: i32) -> i64 "None" {
    %c0 = spv.Constant 0 : i32

    %addr = spv.mlir.addressof @var01_i64 : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=8> [0])>, StorageBuffer>
    %ac = spv.AccessChain %addr[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<i64, stride=8> [0])>, StorageBuffer>, i32, i32
    %val = spv.Load "StorageBuffer" %ac : i64

    spv.ReturnValue %val : i64
  }
}

// CHECK-LABEL: spv.module

// CHECK:  spv.func @load_mixed_scalar_vector_primitive_types(%[[IDX:.+]]: i32)

// CHECK:    %[[ADDR:.+]] = spv.mlir.addressof @var01_v2f2
// CHECK:    %[[ONE:.+]] = spv.Constant 1 : i32
// CHECK:    %[[DIV:.+]] = spv.SDiv %[[IDX]], %[[ONE]] : i32
// CHECK:    %[[MOD:.+]] = spv.SMod %[[IDX]], %[[ONE]] : i32
// CHECK:    spv.AccessChain %[[ADDR]][%{{.+}}, %[[DIV]], %[[MOD]]]
// CHECK:    spv.Load
// CHECK:    spv.Load

// -----

spv.module Logical GLSL450 {
  spv.GlobalVariable @var01_v2f2 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<vector<2xf32>, stride=16> [0])>, StorageBuffer>
  spv.GlobalVariable @var01_i16 bind(0, 1) {aliased} : !spv.ptr<!spv.struct<(!spv.rtarray<i16, stride=2> [0])>, StorageBuffer>

  spv.func @scalar_type_bitwidth_smaller_than_vector(%i0: i32) -> i16 "None" {
    %c0 = spv.Constant 0 : i32

    %addr = spv.mlir.addressof @var01_i16 : !spv.ptr<!spv.struct<(!spv.rtarray<i16, stride=2> [0])>, StorageBuffer>
    %ac = spv.AccessChain %addr[%c0, %i0] : !spv.ptr<!spv.struct<(!spv.rtarray<i16, stride=2> [0])>, StorageBuffer>, i32, i32
    %val = spv.Load "StorageBuffer" %ac : i16

    spv.ReturnValue %val : i16
  }
}

// CHECK-LABEL: spv.module

// CHECK: spv.GlobalVariable @var01_v2f2 bind(0, 1) {aliased}
// CHECK: spv.GlobalVariable @var01_i16 bind(0, 1) {aliased}

// CHECK: spv.func @scalar_type_bitwidth_smaller_than_vector
