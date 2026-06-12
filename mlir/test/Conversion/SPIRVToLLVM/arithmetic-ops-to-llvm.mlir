// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.IAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iadd_scalar
spirv.func @iadd_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.add %{{.*}}, %{{.*}} : i32
  %0 = spirv.IAdd %arg0, %arg1 : i32
  spirv.Return
}

// CHECK-LABEL: @iadd_vector
spirv.func @iadd_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) "None" {
  // CHECK: llvm.add %{{.*}}, %{{.*}} : vector<4xi64>
  %0 = spirv.IAdd %arg0, %arg1 : vector<4xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.IAddCarry
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iaddcarry_scalar
spirv.func @iaddcarry_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: %[[O:.*]] = "llvm.intr.uadd.with.overflow"(%{{.*}}, %{{.*}}) : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: %[[LOW:.*]] = llvm.extractvalue %[[O]][0] : !llvm.struct<(i32, i1)>
  // CHECK: %[[CARRY:.*]] = llvm.extractvalue %[[O]][1] : !llvm.struct<(i32, i1)>
  // CHECK: %[[CARRYEXT:.*]] = llvm.zext %[[CARRY]] : i1 to i32
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.poison : !llvm.struct<packed (i32, i32)>
  // CHECK: %[[R0:.*]] = llvm.insertvalue %[[LOW]], %[[UNDEF]][0] : !llvm.struct<packed (i32, i32)>
  // CHECK: llvm.insertvalue %[[CARRYEXT]], %[[R0]][1] : !llvm.struct<packed (i32, i32)>
  %0 = spirv.IAddCarry %arg0, %arg1 : !spirv.struct<(i32, i32)>
  spirv.Return
}

// CHECK-LABEL: @iaddcarry_vector
spirv.func @iaddcarry_vector(%arg0: vector<2xi32>, %arg1: vector<2xi32>) "None" {
  // CHECK: %[[O:.*]] = "llvm.intr.uadd.with.overflow"(%{{.*}}, %{{.*}}) : (vector<2xi32>, vector<2xi32>) -> !llvm.struct<(vector<2xi32>, vector<2xi1>)>
  // CHECK: %[[LOW:.*]] = llvm.extractvalue %[[O]][0] : !llvm.struct<(vector<2xi32>, vector<2xi1>)>
  // CHECK: %[[CARRY:.*]] = llvm.extractvalue %[[O]][1] : !llvm.struct<(vector<2xi32>, vector<2xi1>)>
  // CHECK: %[[CARRYEXT:.*]] = llvm.zext %[[CARRY]] : vector<2xi1> to vector<2xi32>
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.poison : !llvm.struct<packed (vector<2xi32>, vector<2xi32>)>
  // CHECK: %[[R0:.*]] = llvm.insertvalue %[[LOW]], %[[UNDEF]][0] : !llvm.struct<packed (vector<2xi32>, vector<2xi32>)>
  // CHECK: llvm.insertvalue %[[CARRYEXT]], %[[R0]][1] : !llvm.struct<packed (vector<2xi32>, vector<2xi32>)>
  %0 = spirv.IAddCarry %arg0, %arg1 : !spirv.struct<(vector<2xi32>, vector<2xi32>)>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ISub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @isub_scalar
spirv.func @isub_scalar(%arg0: i8, %arg1: i8) "None" {
  // CHECK: llvm.sub %{{.*}}, %{{.*}} : i8
  %0 = spirv.ISub %arg0, %arg1 : i8
  spirv.Return
}

// CHECK-LABEL: @isub_vector
spirv.func @isub_vector(%arg0: vector<2xi16>, %arg1: vector<2xi16>) "None" {
  // CHECK: llvm.sub %{{.*}}, %{{.*}} : vector<2xi16>
  %0 = spirv.ISub %arg0, %arg1 : vector<2xi16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ISubBorrow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @isubborrow_scalar
spirv.func @isubborrow_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: %[[O:.*]] = "llvm.intr.usub.with.overflow"(%{{.*}}, %{{.*}}) : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: %[[LOW:.*]] = llvm.extractvalue %[[O]][0] : !llvm.struct<(i32, i1)>
  // CHECK: %[[BORROW:.*]] = llvm.extractvalue %[[O]][1] : !llvm.struct<(i32, i1)>
  // CHECK: %[[BORROWEXT:.*]] = llvm.zext %[[BORROW]] : i1 to i32
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.poison : !llvm.struct<packed (i32, i32)>
  // CHECK: %[[R0:.*]] = llvm.insertvalue %[[LOW]], %[[UNDEF]][0] : !llvm.struct<packed (i32, i32)>
  // CHECK: llvm.insertvalue %[[BORROWEXT]], %[[R0]][1] : !llvm.struct<packed (i32, i32)>
  %0 = spirv.ISubBorrow %arg0, %arg1 : !spirv.struct<(i32, i32)>
  spirv.Return
}

// CHECK-LABEL: @isubborrow_vector
spirv.func @isubborrow_vector(%arg0: vector<2xi32>, %arg1: vector<2xi32>) "None" {
  // CHECK: %[[O:.*]] = "llvm.intr.usub.with.overflow"(%{{.*}}, %{{.*}}) : (vector<2xi32>, vector<2xi32>) -> !llvm.struct<(vector<2xi32>, vector<2xi1>)>
  // CHECK: %[[LOW:.*]] = llvm.extractvalue %[[O]][0] : !llvm.struct<(vector<2xi32>, vector<2xi1>)>
  // CHECK: %[[BORROW:.*]] = llvm.extractvalue %[[O]][1] : !llvm.struct<(vector<2xi32>, vector<2xi1>)>
  // CHECK: %[[BORROWEXT:.*]] = llvm.zext %[[BORROW]] : vector<2xi1> to vector<2xi32>
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.poison : !llvm.struct<packed (vector<2xi32>, vector<2xi32>)>
  // CHECK: %[[R0:.*]] = llvm.insertvalue %[[LOW]], %[[UNDEF]][0] : !llvm.struct<packed (vector<2xi32>, vector<2xi32>)>
  // CHECK: llvm.insertvalue %[[BORROWEXT]], %[[R0]][1] : !llvm.struct<packed (vector<2xi32>, vector<2xi32>)>
  %0 = spirv.ISubBorrow %arg0, %arg1 : !spirv.struct<(vector<2xi32>, vector<2xi32>)>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.IMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @imul_scalar
spirv.func @imul_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.mul %{{.*}}, %{{.*}} : i32
  %0 = spirv.IMul %arg0, %arg1 : i32
  spirv.Return
}

// CHECK-LABEL: @imul_vector
spirv.func @imul_vector(%arg0: vector<3xi32>, %arg1: vector<3xi32>) "None" {
  // CHECK: llvm.mul %{{.*}}, %{{.*}} : vector<3xi32>
  %0 = spirv.IMul %arg0, %arg1 : vector<3xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FAdd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fadd_scalar
spirv.func @fadd_scalar(%arg0: f16, %arg1: f16) "None" {
  // CHECK: llvm.fadd %{{.*}}, %{{.*}} : f16
  %0 = spirv.FAdd %arg0, %arg1 : f16
  spirv.Return
}

// CHECK-LABEL: @fadd_vector
spirv.func @fadd_vector(%arg0: vector<4xf32>, %arg1: vector<4xf32>) "None" {
  // CHECK: llvm.fadd %{{.*}}, %{{.*}} : vector<4xf32>
  %0 = spirv.FAdd %arg0, %arg1 : vector<4xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FSub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fsub_scalar
spirv.func @fsub_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fsub %{{.*}}, %{{.*}} : f32
  %0 = spirv.FSub %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @fsub_vector
spirv.func @fsub_vector(%arg0: vector<2xf32>, %arg1: vector<2xf32>) "None" {
  // CHECK: llvm.fsub %{{.*}}, %{{.*}} : vector<2xf32>
  %0 = spirv.FSub %arg0, %arg1 : vector<2xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fdiv_scalar
spirv.func @fdiv_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : f32
  %0 = spirv.FDiv %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @fdiv_vector
spirv.func @fdiv_vector(%arg0: vector<3xf64>, %arg1: vector<3xf64>) "None" {
  // CHECK: llvm.fdiv %{{.*}}, %{{.*}} : vector<3xf64>
  %0 = spirv.FDiv %arg0, %arg1 : vector<3xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fmul_scalar
spirv.func @fmul_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.fmul %{{.*}}, %{{.*}} : f32
  %0 = spirv.FMul %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @fmul_vector
spirv.func @fmul_vector(%arg0: vector<2xf32>, %arg1: vector<2xf32>) "None" {
  // CHECK: llvm.fmul %{{.*}}, %{{.*}} : vector<2xf32>
  %0 = spirv.FMul %arg0, %arg1 : vector<2xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FRem
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @frem_scalar
spirv.func @frem_scalar(%arg0: f32, %arg1: f32) "None" {
  // CHECK: llvm.frem %{{.*}}, %{{.*}} : f32
  %0 = spirv.FRem %arg0, %arg1 : f32
  spirv.Return
}

// CHECK-LABEL: @frem_vector
spirv.func @frem_vector(%arg0: vector<3xf64>, %arg1: vector<3xf64>) "None" {
  // CHECK: llvm.frem %{{.*}}, %{{.*}} : vector<3xf64>
  %0 = spirv.FRem %arg0, %arg1 : vector<3xf64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.FNegate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fneg_scalar
spirv.func @fneg_scalar(%arg: f64) "None" {
  // CHECK: llvm.fneg %{{.*}} : f64
  %0 = spirv.FNegate %arg : f64
  spirv.Return
}

// CHECK-LABEL: @fneg_vector
spirv.func @fneg_vector(%arg: vector<2xf32>) "None" {
  // CHECK: llvm.fneg %{{.*}} : vector<2xf32>
  %0 = spirv.FNegate %arg : vector<2xf32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.UDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @udiv_scalar
spirv.func @udiv_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.udiv %{{.*}}, %{{.*}} : i32
  %0 = spirv.UDiv %arg0, %arg1 : i32
  spirv.Return
}

// CHECK-LABEL: @udiv_vector
spirv.func @udiv_vector(%arg0: vector<3xi64>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.udiv %{{.*}}, %{{.*}} : vector<3xi64>
  %0 = spirv.UDiv %arg0, %arg1 : vector<3xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.UMod
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @umod_scalar
spirv.func @umod_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.urem %{{.*}}, %{{.*}} : i32
  %0 = spirv.UMod %arg0, %arg1 : i32
  spirv.Return
}

// CHECK-LABEL: @umod_vector
spirv.func @umod_vector(%arg0: vector<3xi64>, %arg1: vector<3xi64>) "None" {
  // CHECK: llvm.urem %{{.*}}, %{{.*}} : vector<3xi64>
  %0 = spirv.UMod %arg0, %arg1 : vector<3xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.SDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sdiv_scalar
spirv.func @sdiv_scalar(%arg0: i16, %arg1: i16) "None" {
  // CHECK: llvm.sdiv %{{.*}}, %{{.*}} : i16
  %0 = spirv.SDiv %arg0, %arg1 : i16
  spirv.Return
}

// CHECK-LABEL: @sdiv_vector
spirv.func @sdiv_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) "None" {
  // CHECK: llvm.sdiv %{{.*}}, %{{.*}} : vector<2xi64>
  %0 = spirv.SDiv %arg0, %arg1 : vector<2xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.SRem
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @srem_scalar
spirv.func @srem_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.srem %{{.*}}, %{{.*}} : i32
  %0 = spirv.SRem %arg0, %arg1 : i32
  spirv.Return
}

// CHECK-LABEL: @srem_vector
spirv.func @srem_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
  // CHECK: llvm.srem %{{.*}}, %{{.*}} : vector<4xi32>
  %0 = spirv.SRem %arg0, %arg1 : vector<4xi32>
  spirv.Return
}
