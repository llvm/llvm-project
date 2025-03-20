// RUN: mlir-opt %s -convert-index-to-spirv | FileCheck %s
// RUN: mlir-opt %s -convert-index-to-spirv=use-64bit-index=false | FileCheck %s --check-prefix=INDEX32
// RUN: mlir-opt %s -convert-index-to-spirv=use-64bit-index=true | FileCheck %s --check-prefix=INDEX64

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Int64], []>, #spirv.resource_limits<>>
} {
// CHECK-LABEL: @trivial_ops
func.func @trivial_ops(%a: index, %b: index) {
  // CHECK: spirv.IAdd
  %0 = index.add %a, %b
  // CHECK: spirv.ISub
  %1 = index.sub %a, %b
  // CHECK: spirv.IMul
  %2 = index.mul %a, %b
  // CHECK: spirv.SDiv
  %3 = index.divs %a, %b
  // CHECK: spirv.UDiv
  %4 = index.divu %a, %b
  // CHECK: spirv.SRem
  %5 = index.rems %a, %b
  // CHECK: spirv.UMod
  %6 = index.remu %a, %b
  // CHECK: spirv.GL.SMax
  %7 = index.maxs %a, %b
  // CHECK: spirv.GL.UMax
  %8 = index.maxu %a, %b
  // CHECK: spirv.GL.SMin
  %9 = index.mins %a, %b
  // CHECK: spirv.GL.UMin
  %10 = index.minu %a, %b
  // CHECK: spirv.ShiftLeftLogical
  %11 = index.shl %a, %b
  // CHECK: spirv.ShiftRightArithmetic
  %12 = index.shrs %a, %b
  // CHECK: spirv.ShiftRightLogical
  %13 = index.shru %a, %b
  return
}

// CHECK-LABEL: @bitwise_ops
func.func @bitwise_ops(%a: index, %b: index) {
  // CHECK: spirv.BitwiseAnd
  %0 = index.and %a, %b
  // CHECK: spirv.BitwiseOr
  %1 = index.or %a, %b
  // CHECK: spirv.BitwiseXor
  %2 = index.xor %a, %b
  return
}

// INDEX32-LABEL: @constant_ops
// INDEX64-LABEL: @constant_ops
func.func @constant_ops() {
  // INDEX32: spirv.Constant 42 : i32
  // INDEX64: spirv.Constant 42 : i64
  %0 = index.constant 42
  // INDEX32: spirv.Constant true
  // INDEX64: spirv.Constant true
  %1 = index.bool.constant true
  // INDEX32: spirv.Constant false
  // INDEX64: spirv.Constant false
  %2 = index.bool.constant false
  return
}

// CHECK-LABEL: @ceildivs
// CHECK-SAME: %[[NI:.*]]: index, %[[MI:.*]]: index
func.func @ceildivs(%n: index, %m: index) -> index {
  // CHECK-DAG: %[[N:.*]] = builtin.unrealized_conversion_cast %[[NI]]
  // CHECK-DAG: %[[M:.*]] = builtin.unrealized_conversion_cast %[[MI]]
  // CHECK: %[[ZERO:.*]] = spirv.Constant 0
  // CHECK: %[[POS_ONE:.*]] = spirv.Constant 1
  // CHECK: %[[NEG_ONE:.*]] = spirv.Constant -1

  // CHECK: %[[M_POS:.*]] = spirv.SGreaterThan %[[M]], %[[ZERO]]
  // CHECK: %[[X:.*]] = spirv.Select %[[M_POS]], %[[NEG_ONE]], %[[POS_ONE]]

  // CHECK: %[[N_PLUS_X:.*]] = spirv.IAdd %[[N]], %[[X]]
  // CHECK: %[[N_PLUS_X_DIV_M:.*]] = spirv.SDiv %[[N_PLUS_X]], %[[M]]
  // CHECK: %[[POS_RES:.*]] = spirv.IAdd %[[N_PLUS_X_DIV_M]], %[[POS_ONE]]

  // CHECK: %[[NEG_N:.*]] = spirv.ISub %[[ZERO]], %[[N]]
  // CHECK: %[[NEG_N_DIV_M:.*]] = spirv.SDiv %[[NEG_N]], %[[M]]
  // CHECK: %[[NEG_RES:.*]] = spirv.ISub %[[ZERO]], %[[NEG_N_DIV_M]]

  // CHECK: %[[N_POS:.*]] = spirv.SGreaterThan %[[N]], %[[ZERO]]
  // CHECK: %[[SAME_SIGN:.*]] = spirv.LogicalEqual %[[N_POS]], %[[M_POS]]
  // CHECK: %[[N_NON_ZERO:.*]] = spirv.INotEqual %[[N]], %[[ZERO]]
  // CHECK: %[[CMP:.*]] = spirv.LogicalAnd %[[SAME_SIGN]], %[[N_NON_ZERO]]
  // CHECK: %[[RESULT:.*]] = spirv.Select %[[CMP]], %[[POS_RES]], %[[NEG_RES]]
  %result = index.ceildivs %n, %m

  // %[[RESULTI:.*] = builtin.unrealized_conversion_cast %[[RESULT]]
  // return %[[RESULTI]]
  return %result : index
}

// CHECK-LABEL: @ceildivu
// CHECK-SAME: %[[NI:.*]]: index, %[[MI:.*]]: index
func.func @ceildivu(%n: index, %m: index) -> index {
  // CHECK-DAG: %[[N:.*]] = builtin.unrealized_conversion_cast %[[NI]]
  // CHECK-DAG: %[[M:.*]] = builtin.unrealized_conversion_cast %[[MI]]
  // CHECK: %[[ZERO:.*]] = spirv.Constant 0
  // CHECK: %[[ONE:.*]] = spirv.Constant 1

  // CHECK: %[[N_MINUS_ONE:.*]] = spirv.ISub %[[N]], %[[ONE]]
  // CHECK: %[[N_MINUS_ONE_DIV_M:.*]] = spirv.UDiv %[[N_MINUS_ONE]], %[[M]]
  // CHECK: %[[N_MINUS_ONE_DIV_M_PLUS_ONE:.*]] = spirv.IAdd %[[N_MINUS_ONE_DIV_M]], %[[ONE]]

  // CHECK: %[[CMP:.*]] = spirv.IEqual %[[N]], %[[ZERO]]
  // CHECK: %[[RESULT:.*]] = spirv.Select %[[CMP]], %[[ZERO]], %[[N_MINUS_ONE_DIV_M_PLUS_ONE]]
  %result = index.ceildivu %n, %m

  // %[[RESULTI:.*] = builtin.unrealized_conversion_cast %[[RESULT]]
  // return %[[RESULTI]]
  return %result : index
}

// CHECK-LABEL: @floordivs
// CHECK-SAME: %[[NI:.*]]: index, %[[MI:.*]]: index
func.func @floordivs(%n: index, %m: index) -> index {
  // CHECK-DAG: %[[N:.*]] = builtin.unrealized_conversion_cast %[[NI]]
  // CHECK-DAG: %[[M:.*]] = builtin.unrealized_conversion_cast %[[MI]]
  // CHECK: %[[ZERO:.*]] = spirv.Constant 0
  // CHECK: %[[POS_ONE:.*]] = spirv.Constant 1
  // CHECK: %[[NEG_ONE:.*]] = spirv.Constant -1

  // CHECK: %[[M_NEG:.*]] = spirv.SLessThan %[[M]], %[[ZERO]]
  // CHECK: %[[X:.*]] = spirv.Select %[[M_NEG]], %[[POS_ONE]], %[[NEG_ONE]]

  // CHECK: %[[X_MINUS_N:.*]] = spirv.ISub %[[X]], %[[N]]
  // CHECK: %[[X_MINUS_N_DIV_M:.*]] = spirv.SDiv %[[X_MINUS_N]], %[[M]]
  // CHECK: %[[NEG_RES:.*]] = spirv.ISub %[[NEG_ONE]], %[[X_MINUS_N_DIV_M]]

  // CHECK: %[[POS_RES:.*]] = spirv.SDiv %[[N]], %[[M]]

  // CHECK: %[[N_NEG:.*]] = spirv.SLessThan %[[N]], %[[ZERO]]
  // CHECK: %[[DIFF_SIGN:.*]] = spirv.LogicalNotEqual %[[N_NEG]], %[[M_NEG]]
  // CHECK: %[[N_NON_ZERO:.*]] = spirv.INotEqual %[[N]], %[[ZERO]]

  // CHECK: %[[CMP:.*]] = spirv.LogicalAnd %[[DIFF_SIGN]], %[[N_NON_ZERO]]
  // CHECK: %[[RESULT:.*]] = spirv.Select %[[CMP]], %[[POS_RES]], %[[NEG_RES]]
  %result = index.floordivs %n, %m

  // %[[RESULTI:.*] = builtin.unrealized_conversion_cast %[[RESULT]]
  // return %[[RESULTI]]
  return %result : index
}

// CHECK-LABEL: @index_cmp
func.func @index_cmp(%a : index, %b : index) {
  // CHECK: spirv.IEqual
  %0 = index.cmp eq(%a, %b)
  // CHECK: spirv.INotEqual
  %1 = index.cmp ne(%a, %b)

  // CHECK: spirv.SLessThan
  %2 = index.cmp slt(%a, %b)
  // CHECK: spirv.SLessThanEqual
  %3 = index.cmp sle(%a, %b)
  // CHECK: spirv.SGreaterThan
  %4 = index.cmp sgt(%a, %b)
  // CHECK: spirv.SGreaterThanEqual
  %5 = index.cmp sge(%a, %b)

  // CHECK: spirv.ULessThan
  %6 = index.cmp ult(%a, %b)
  // CHECK: spirv.ULessThanEqual
  %7 = index.cmp ule(%a, %b)
  // CHECK: spirv.UGreaterThan
  %8 = index.cmp ugt(%a, %b)
  // CHECK: spirv.UGreaterThanEqual
  %9 = index.cmp uge(%a, %b)
  return
}

// CHECK-LABEL: @index_sizeof
func.func @index_sizeof() {
  // CHECK: spirv.Constant 32 : i32
  %0 = index.sizeof
  return
}

// INDEX32-LABEL: @index_cast_from
// INDEX64-LABEL: @index_cast_from
// INDEX32-SAME: %[[AI:.*]]: index
// INDEX64-SAME: %[[AI:.*]]: index
func.func @index_cast_from(%a: index) -> (i64, i32, i64, i32) {
  // INDEX32: %[[A:.*]] = builtin.unrealized_conversion_cast %[[AI]] : index to i32
  // INDEX64: %[[A:.*]] = builtin.unrealized_conversion_cast %[[AI]] : index to i64

  // INDEX32: %[[V0:.*]] = spirv.SConvert %[[A]] : i32 to i64
  %0 = index.casts %a : index to i64
  // INDEX64: %[[V1:.*]] = spirv.SConvert %[[A]] : i64 to i32
  %1 = index.casts %a : index to i32
  // INDEX32: %[[V2:.*]] = spirv.UConvert %[[A]] : i32 to i64
  %2 = index.castu %a : index to i64
  // INDEX64: %[[V3:.*]] = spirv.UConvert %[[A]] : i64 to i32
  %3 = index.castu %a : index to i32

  // INDEX32: return %[[V0]], %[[A]], %[[V2]], %[[A]]
  // INDEX64: return %[[A]], %[[V1]], %[[A]], %[[V3]]
  return %0, %1, %2, %3 : i64, i32, i64, i32
}

// INDEX32-LABEL: @index_cast_to
// INDEX64-LABEL: @index_cast_to
// INDEX32-SAME: %[[A:.*]]: i32, %[[B:.*]]: i64
// INDEX64-SAME: %[[A:.*]]: i32, %[[B:.*]]: i64
func.func @index_cast_to(%a: i32, %b: i64) -> (index, index, index, index) {
  // INDEX64: %[[V0:.*]] = spirv.SConvert %[[A]] : i32 to i64
  %0 = index.casts %a : i32 to index
  // INDEX32: %[[V1:.*]] = spirv.SConvert %[[B]] : i64 to i32
  %1 = index.casts %b : i64 to index
  // INDEX64: %[[V2:.*]] = spirv.UConvert %[[A]] : i32 to i64
  %2 = index.castu %a : i32 to index
  // INDEX32: %[[V3:.*]] = spirv.UConvert %[[B]] : i64 to i32
  %3 = index.castu %b : i64 to index
  return %0, %1, %2, %3 : index, index, index, index
}
}
