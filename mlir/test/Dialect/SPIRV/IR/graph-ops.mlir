// RUN: mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ARM.Graph and spirv.ARM.GraphOutputs
//===----------------------------------------------------------------------===//

// CHECK: spirv.ARM.Graph {{@.*}}({{%.*}}: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
spirv.ARM.Graph @graphAndOutputs(%arg0: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<14x19xi16>
  spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<14x19xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.GraphConstant
//===----------------------------------------------------------------------===//

// CHECK: spirv.ARM.Graph {{@.*}}() -> !spirv.arm.tensor<2x3xi16> {
spirv.ARM.Graph @graphConstant() -> !spirv.arm.tensor<2x3xi16> {
  // CHECK: [[CONST:%.*]] = spirv.ARM.GraphConstant {graph_constant_id = 42 : i32} : !spirv.arm.tensor<2x3xi16>
  %0 = spirv.ARM.GraphConstant { graph_constant_id = 42 : i32 } : !spirv.arm.tensor<2x3xi16>
  // CHECK: spirv.ARM.GraphOutputs [[CONST:%.*]] : !spirv.arm.tensor<2x3xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x3xi16>
}
// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.GraphEntryPoint
//===----------------------------------------------------------------------===//

// CHECK: spirv.GlobalVariable [[VARARG0:@.*]] bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
spirv.GlobalVariable @entrypoint_arg_0 bind(0, 0) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
// CHECK: spirv.GlobalVariable [[VARRES0:@.*]] bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
spirv.GlobalVariable @entrypoint_res_0 bind(0, 1) : !spirv.ptr<!spirv.arm.tensor<14x19xi16>, UniformConstant>
// CHECK: spirv.ARM.GraphEntryPoint [[GN:@.*]], [[VARARG0]], [[VARRES0]]
spirv.ARM.GraphEntryPoint @entrypoint, @entrypoint_arg_0, @entrypoint_res_0

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.Graph with no terminator
//===----------------------------------------------------------------------===//

// expected-error @+1 {{empty block: expect at least a terminator}}
spirv.ARM.Graph @graphNoterminator(%arg0: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.Graph with no result types
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spirv.ARM.Graph' op there should be at least one result}}
spirv.ARM.Graph @graphNoOutputs(%arg0: !spirv.arm.tensor<14x19xi16>) -> () {
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.GraphConstant outside graph scope
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spirv.ARM.GraphConstant' op failed to verify that op must appear in a spirv.ARM.Graph op's block}}
%0 = spirv.ARM.GraphConstant { graph_constant_id = 42 : i32 } : !spirv.arm.tensor<2x3xi16>
// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.GraphOutputs outside graph scope
//===----------------------------------------------------------------------===//

%0 = spirv.Constant dense<1> : !spirv.arm.tensor<1xi16>
// expected-error @+1 {{'spirv.ARM.GraphOutputs' op failed to verify that op must appear in a spirv.ARM.Graph op's block}}
spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<1xi16>

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.Graph return type does not match spirv.ARM.GraphOutputs
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @graphAndOutputs(%arg0: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<5x3xi16> {
  // expected-error @+1 {{type of return operand 0 ('!spirv.arm.tensor<14x19xi16>') doesn't match graph result type ('!spirv.arm.tensor<5x3xi16>')}}
  spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<14x19xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.Graph return type does not match number of results in spirv.ARM.GraphOutputs
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @graphAndOutputs(%arg0: !spirv.arm.tensor<14x19xi16>) -> (!spirv.arm.tensor<14x19xi16>, !spirv.arm.tensor<14x19xi16>) {
  // expected-error @+1 {{'spirv.ARM.GraphOutputs' op is returning 1 value(s) but enclosing spirv.ARM.Graph requires 2 result(s)}}
  spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<14x19xi16>
}

// -----

spirv.ARM.Graph @graphAndOutputs(%arg0: !spirv.arm.tensor<14x19xi16>) -> !spirv.arm.tensor<14x19xi16> {
  // expected-error @+1 {{'spirv.ARM.GraphOutputs' op is returning 2 value(s) but enclosing spirv.ARM.Graph requires 1 result(s)}}
  spirv.ARM.GraphOutputs %arg0, %arg0 : !spirv.arm.tensor<14x19xi16>, !spirv.arm.tensor<14x19xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.Graph using a non TensorArmType argument
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spirv.ARM.Graph' op type of argument #0 must be a TensorArmType, but got 'i8'}}
spirv.ARM.Graph @graphAndOutputs(%arg0: i8) -> !spirv.arm.tensor<14x19xi16> {
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ARM.Graph using a non TensorArmType result
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spirv.ARM.Graph' op type of result #0 must be a TensorArmType, but got 'i8'}}
spirv.ARM.Graph @graphAndOutputs(%arg0: !spirv.arm.tensor<14x19xi16>) -> i8 {
}
