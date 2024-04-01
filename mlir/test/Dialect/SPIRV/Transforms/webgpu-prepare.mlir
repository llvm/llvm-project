// RUN: mlir-opt --split-input-file --verify-diagnostics \
// RUN:   --spirv-webgpu-prepare --cse %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.UMulExtended
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {

// CHECK-LABEL: func @umul_extended_i32
// CHECK-SAME:       ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant 65535 : i32
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant 16 : i32
// CHECK-NEXT:       [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : i32
// CHECK-NEXT:       [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : i32
// CHECK-NEXT:       [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : i32
// CHECK-NEXT:       [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : i32
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSHI]]
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]] : i32
// CHECK:                              spirv.BitwiseOr
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]] : i32
// CHECK:                              spirv.BitwiseOr
// CHECK:            [[RES:%.+]]     = spirv.CompositeConstruct [[RESLO:%.+]], [[RESHI:%.+]] : (i32, i32) -> !spirv.struct<(i32, i32)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @umul_extended_i32(%arg0 : i32, %arg1 : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.UMulExtended %arg0, %arg1 : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @umul_extended_vector_i32
// CHECK-SAME:       ([[ARG0:%.+]]: vector<3xi32>, [[ARG1:%.+]]: vector<3xi32>)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant dense<65535> : vector<3xi32>
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant dense<16> : vector<3xi32>
// CHECK-NEXT:       [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : vector<3xi32>
// CHECK-NEXT:       [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : vector<3xi32>
// CHECK-NEXT:       [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : vector<3xi32>
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSHI]]
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK-DAG:                          spirv.IAdd
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]]
// CHECK:                              spirv.BitwiseOr
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]]
// CHECK:                              spirv.BitwiseOr
// CHECK-NEXT:       [[RES:%.+]]     = spirv.CompositeConstruct [[RESLOW:%.+]], [[RESHI:%.+]]
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @umul_extended_vector_i32(%arg0 : vector<3xi32>, %arg1 : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.UMulExtended %arg0, %arg1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @umul_extended_i16
// CHECK-NEXT:       spirv.UMulExtended
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @umul_extended_i16(%arg : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.UMulExtended %arg, %arg : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

//===----------------------------------------------------------------------===//
// spirv.SMulExtended
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @smul_extended_i32
// CHECK-SAME:       ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant 65535 : i32
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant 16 : i32
// CHECK-NEXT:       [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : i32
// CHECK-NEXT:       [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : i32
// CHECK-NEXT:       [[LHSSIGN:%.+]] = spirv.ShiftRightArithmetic [[ARG0]], [[CST16]] : i32
// CHECK-NEXT:       [[LHSEXT:%.+]]  = spirv.ShiftRightLogical [[LHSSIGN]], [[CST16]] : i32
// CHECK-NEXT:       [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : i32
// CHECK-NEXT:       [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : i32
// CHECK-NEXT:       [[RHSSIGN:%.+]] = spirv.ShiftRightArithmetic [[ARG1]], [[CST16]] : i32
// CHECK-NEXT:       [[RHSEXT:%.+]]  = spirv.ShiftRightLogical [[RHSSIGN]], [[CST16]] : i32
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSEXT]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSEXT]]
// CHECK-DAG:                          spirv.IMul [[LHSEXT]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSEXT]], [[RHSHI]]
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]] : i32
// CHECK:                              spirv.BitwiseOr
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]] : i32
// CHECK:                              spirv.BitwiseOr
// CHECK:            [[RES:%.+]]     = spirv.CompositeConstruct [[RESLO:%.+]], [[RESHI:%.+]] : (i32, i32) -> !spirv.struct<(i32, i32)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @smul_extended_i32(%arg0 : i32, %arg1 : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.SMulExtended %arg0, %arg1 : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @smul_extended_vector_i32
// CHECK-SAME:       ([[ARG0:%.+]]: vector<3xi32>, [[ARG1:%.+]]: vector<3xi32>)
// CHECK-DAG:        [[CSTMASK:%.+]] = spirv.Constant dense<65535> : vector<3xi32>
// CHECK-DAG:        [[CST16:%.+]]   = spirv.Constant dense<16> : vector<3xi32>
// CHECK-NEXT:       [[LHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG0]], [[CSTMASK]] : vector<3xi32>
// CHECK-NEXT:       [[LHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG0]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[LHSSIGN:%.+]] = spirv.ShiftRightArithmetic [[ARG0]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[LHSEXT:%.+]]  = spirv.ShiftRightLogical [[LHSSIGN]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[RHSLOW:%.+]]  = spirv.BitwiseAnd [[ARG1]], [[CSTMASK]] : vector<3xi32>
// CHECK-NEXT:       [[RHSHI:%.+]]   = spirv.ShiftRightLogical [[ARG1]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[RHSSIGN:%.+]] = spirv.ShiftRightArithmetic [[ARG1]], [[CST16]] : vector<3xi32>
// CHECK-NEXT:       [[RHSEXT:%.+]]  = spirv.ShiftRightLogical [[RHSSIGN]], [[CST16]] : vector<3xi32>
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSLOW]], [[RHSEXT]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSHI]]
// CHECK-DAG:                          spirv.IMul [[LHSHI]],  [[RHSEXT]]
// CHECK-DAG:                          spirv.IMul [[LHSEXT]], [[RHSLOW]]
// CHECK-DAG:                          spirv.IMul [[LHSEXT]], [[RHSHI]]
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]]
// CHECK:                              spirv.BitwiseOr
// CHECK:                              spirv.ShiftLeftLogical {{%.+}}, [[CST16]]
// CHECK:                              spirv.BitwiseOr
// CHECK-NEXT:       [[RES:%.+]]     = spirv.CompositeConstruct [[RESLOW:%.+]], [[RESHI:%.+]]
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @smul_extended_vector_i32(%arg0 : vector<3xi32>, %arg1 : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.SMulExtended %arg0, %arg1 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @smul_extended_i16
// CHECK-NEXT:       spirv.SMulExtended
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @smul_extended_i16(%arg : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.SMulExtended %arg, %arg : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

// CHECK-LABEL: func @iaddcarry_i32
// CHECK-SAME:       ([[A:%.+]]: i32, [[B:%.+]]: i32)
// CHECK-NEXT:       [[ONE:%.+]]    = spirv.Constant 1 : i32
// CHECK-NEXT:       [[ZERO:%.+]]   = spirv.Constant 0 : i32
// CHECK-NEXT:       [[OUT:%.+]]    = spirv.IAdd [[A]], [[B]]
// CHECK-NEXT:       [[CMP:%.+]]    = spirv.ULessThan [[OUT]], [[A]]
// CHECK-NEXT:       [[CARRY:%.+]]  = spirv.Select [[CMP]], [[ONE]], [[ZERO]]
// CHECK-NEXT:       [[RES:%.+]]     = spirv.CompositeConstruct [[OUT]], [[CARRY]] : (i32, i32) -> !spirv.struct<(i32, i32)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(i32, i32)>
spirv.func @iaddcarry_i32(%a : i32, %b : i32) -> !spirv.struct<(i32, i32)> "None" {
  %0 = spirv.IAddCarry %a, %b : !spirv.struct<(i32, i32)>
  spirv.ReturnValue %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: func @iaddcarry_vector_i32
// CHECK-SAME:       ([[A:%.+]]: vector<3xi32>, [[B:%.+]]: vector<3xi32>)
// CHECK-NEXT:       [[ONE:%.+]]    = spirv.Constant dense<1> : vector<3xi32>
// CHECK-NEXT:       [[ZERO:%.+]]   = spirv.Constant dense<0> : vector<3xi32>
// CHECK-NEXT:       [[OUT:%.+]]    = spirv.IAdd [[A]], [[B]]
// CHECK-NEXT:       [[CMP:%.+]]    = spirv.ULessThan [[OUT]], [[A]]
// CHECK-NEXT:       [[CARRY:%.+]]  = spirv.Select [[CMP]], [[ONE]], [[ZERO]]
// CHECK-NEXT:       [[RES:%.+]]    = spirv.CompositeConstruct [[OUT]], [[CARRY]] : (vector<3xi32>, vector<3xi32>) -> !spirv.struct<(vector<3xi32>, vector<3xi32>)>
// CHECK-NEXT:       spirv.ReturnValue [[RES]] : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
spirv.func @iaddcarry_vector_i32(%a : vector<3xi32>, %b : vector<3xi32>)
  -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> "None" {
  %0 = spirv.IAddCarry %a, %b : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  spirv.ReturnValue %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// CHECK-LABEL: func @iaddcarry_i16
// CHECK-NEXT:       spirv.IAddCarry
// CHECK-NEXT:       spirv.ReturnValue
spirv.func @iaddcarry_i16(%a : i16, %b : i16) -> !spirv.struct<(i16, i16)> "None" {
  %0 = spirv.IAddCarry %a, %b : !spirv.struct<(i16, i16)>
  spirv.ReturnValue %0 : !spirv.struct<(i16, i16)>
}

// CHECK-LABEL: func @is_inf_f32
// CHECK-NEXT:       [[FALSE:%.+]] = spirv.Constant false
// CHECK-NEXT:       spirv.ReturnValue [[FALSE]] : i1
spirv.func @is_inf_f32(%a : f32) -> i1 "None" {
  %0 = spirv.IsInf %a : f32
  spirv.ReturnValue %0 : i1
}

// CHECK-LABEL: func @is_inf_4xf32
// CHECK-NEXT:       [[FALSE:%.+]] = spirv.Constant dense<false> : vector<4xi1>
// CHECK-NEXT:       spirv.ReturnValue [[FALSE]] : vector<4xi1>
spirv.func @is_inf_4xf32(%a : vector<4xf32>) -> vector<4xi1> "None" {
  %0 = spirv.IsInf %a : vector<4xf32>
  spirv.ReturnValue %0 : vector<4xi1>
}

// CHECK-LABEL: func @is_nan_f32
// CHECK-NEXT:       [[FALSE:%.+]] = spirv.Constant false
// CHECK-NEXT:       spirv.ReturnValue [[FALSE]] : i1
spirv.func @is_nan_f32(%a : f32) -> i1 "None" {
  %0 = spirv.IsNan %a : f32
  spirv.ReturnValue %0 : i1
}

// CHECK-LABEL: func @is_nan_4xf32
// CHECK-NEXT:       [[FALSE:%.+]] = spirv.Constant dense<false> : vector<4xi1>
// CHECK-NEXT:       spirv.ReturnValue [[FALSE]] : vector<4xi1>
spirv.func @is_nan_4xf32(%a : vector<4xf32>) -> vector<4xi1> "None" {
  %0 = spirv.IsNan %a : vector<4xf32>
  spirv.ReturnValue %0 : vector<4xi1>
}

} // end module
