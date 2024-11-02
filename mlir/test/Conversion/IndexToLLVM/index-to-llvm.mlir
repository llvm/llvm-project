// RUN: mlir-opt %s -convert-index-to-llvm | FileCheck %s
// RUN: mlir-opt %s -convert-index-to-llvm=index-bitwidth=32 | FileCheck %s --check-prefix=INDEX32
// RUN: mlir-opt %s -convert-index-to-llvm=index-bitwidth=64 | FileCheck %s --check-prefix=INDEX64

// CHECK-LABEL: @trivial_ops
func.func @trivial_ops(%a: index, %b: index) {
  // CHECK: llvm.add
  %0 = index.add %a, %b
  // CHECK: llvm.sub
  %1 = index.sub %a, %b
  // CHECK: llvm.mul
  %2 = index.mul %a, %b
  // CHECK: llvm.sdiv
  %3 = index.divs %a, %b
  // CHECK: llvm.udiv
  %4 = index.divu %a, %b
  // CHECK: llvm.srem
  %5 = index.rems %a, %b
  // CHECK: llvm.urem
  %6 = index.remu %a, %b
  // CHECK: llvm.intr.smax
  %7 = index.maxs %a, %b
  // CHECK: llvm.intr.umax
  %8 = index.maxu %a, %b
  // CHECK: llvm.shl
  %9 = index.shl %a, %b
  // CHECK: llvm.ashr
  %10 = index.shrs %a, %b
  // CHECK: llvm.lshr
  %11 = index.shru %a, %b
  // CHECK: llvm.add
  %12 = index.add %a, %b
  // CHECK: llvm.or
  %13 = index.or %a, %b
  // CHECK: llvm.xor
  %14 = index.xor %a, %b
  // CHECK: llvm.mlir.constant(true
  %15 = index.bool.constant true
  return
}

// CHECK-LABEL: @ceildivs
// CHECK-SAME: %[[NI:.*]]: index, %[[MI:.*]]: index
func.func @ceildivs(%n: index, %m: index) -> index {
  // CHECK: %[[N:.*]] = builtin.unrealized_conversion_cast %[[NI]]
  // CHECK: %[[M:.*]] = builtin.unrealized_conversion_cast %[[MI]]
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[POS_ONE:.*]] = llvm.mlir.constant(1 :
  // CHECK: %[[NEG_ONE:.*]] = llvm.mlir.constant(-1 :

  // CHECK: %[[M_POS:.*]] = llvm.icmp "sgt" %[[M]], %[[ZERO]]
  // CHECK: %[[X:.*]] = llvm.select %[[M_POS]], %[[NEG_ONE]], %[[POS_ONE]]

  // CHECK: %[[N_PLUS_X:.*]] = llvm.add %[[N]], %[[X]]
  // CHECK: %[[N_PLUS_X_DIV_M:.*]] = llvm.sdiv %[[N_PLUS_X]], %[[M]]
  // CHECK: %[[POS_RES:.*]] = llvm.add %[[N_PLUS_X_DIV_M]], %[[POS_ONE]]

  // CHECK: %[[NEG_N:.*]] = llvm.sub %[[ZERO]], %[[N]]
  // CHECK: %[[NEG_N_DIV_M:.*]] = llvm.sdiv %[[NEG_N]], %[[M]]
  // CHECK: %[[NEG_RES:.*]] = llvm.sub %[[ZERO]], %[[NEG_N_DIV_M]]

  // CHECK: %[[N_POS:.*]] = llvm.icmp "sgt" %[[N]], %[[ZERO]]
  // CHECK: %[[SAME_SIGN:.*]] = llvm.icmp "eq" %[[N_POS]], %[[M_POS]]
  // CHECK: %[[N_NON_ZERO:.*]] = llvm.icmp "ne" %[[N]], %[[ZERO]]
  // CHECK: %[[CMP:.*]] = llvm.and %[[SAME_SIGN]], %[[N_NON_ZERO]]
  // CHECK: %[[RESULT:.*]] = llvm.select %[[CMP]], %[[POS_RES]], %[[NEG_RES]]
  %result = index.ceildivs %n, %m

  // CHECK: %[[RESULTI:.*]] = builtin.unrealized_conversion_cast %[[RESULT]]
  // CHECK: return %[[RESULTI]]
  return %result : index
}

// CHECK-LABEL: @ceildivu
// CHECK-SAME: %[[NI:.*]]: index, %[[MI:.*]]: index
func.func @ceildivu(%n: index, %m: index) -> index {
  // CHECK: %[[N:.*]] = builtin.unrealized_conversion_cast %[[NI]]
  // CHECK: %[[M:.*]] = builtin.unrealized_conversion_cast %[[MI]]
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 :

  // CHECK: %[[MINUS_ONE:.*]] = llvm.sub %[[N]], %[[ONE]]
  // CHECK: %[[QUOTIENT:.*]] = llvm.udiv %[[MINUS_ONE]], %[[M]]
  // CHECK: %[[PLUS_ONE:.*]] = llvm.add %[[QUOTIENT]], %[[ONE]]

  // CHECK: %[[CMP:.*]] = llvm.icmp "eq" %[[N]], %[[ZERO]]
  // CHECK: %[[RESULT:.*]] = llvm.select %[[CMP]], %[[ZERO]], %[[PLUS_ONE]]
  %result = index.ceildivu %n, %m

  // CHECK: %[[RESULTI:.*]] = builtin.unrealized_conversion_cast %[[RESULT]]
  // CHECK: return %[[RESULTI]]
  return %result : index
}

// CHECK-LABEL: @floordivs
// CHECK-SAME: %[[NI:.*]]: index, %[[MI:.*]]: index
func.func @floordivs(%n: index, %m: index) -> index {
  // CHECK: %[[N:.*]] = builtin.unrealized_conversion_cast %[[NI]]
  // CHECK: %[[M:.*]] = builtin.unrealized_conversion_cast %[[MI]]
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[POS_ONE:.*]] = llvm.mlir.constant(1 :
  // CHECK: %[[NEG_ONE:.*]] = llvm.mlir.constant(-1 :

  // CHECK: %[[M_NEG:.*]] = llvm.icmp "slt" %[[M]], %[[ZERO]]
  // CHECK: %[[X:.*]] = llvm.select %[[M_NEG]], %[[POS_ONE]], %[[NEG_ONE]]

  // CHECK: %[[X_MINUS_N:.*]] = llvm.sub %[[X]], %[[N]]
  // CHECK: %[[X_MINUS_N_DIV_M:.*]] = llvm.sdiv %[[X_MINUS_N]], %[[M]]
  // CHECK: %[[NEG_RES:.*]] = llvm.sub %[[NEG_ONE]], %[[X_MINUS_N_DIV_M]]

  // CHECK: %[[POS_RES:.*]] = llvm.sdiv %[[N]], %[[M]]

  // CHECK: %[[N_NEG:.*]] = llvm.icmp "slt" %[[N]], %[[ZERO]]
  // CHECK: %[[DIFF_SIGN:.*]] = llvm.icmp "ne" %[[N_NEG]], %[[M_NEG]]
  // CHECK: %[[N_NON_ZERO:.*]] = llvm.icmp "ne" %[[N]], %[[ZERO]]
  // CHECK: %[[CMP:.*]] = llvm.and %[[DIFF_SIGN]], %[[N_NON_ZERO]]
  // CHECK: %[[RESULT:.*]] = llvm.select %[[CMP]], %[[NEG_RES]], %[[POS_RES]]
  %result = index.floordivs %n, %m

  // CHECK: %[[RESULTI:.*]] = builtin.unrealized_conversion_cast %[[RESULT]]
  // CHECK: return %[[RESULTI]]
  return %result : index
}

// INDEX32-LABEL: @index_cast_from
// INDEX64-LABEL: @index_cast_from
// INDEX32-SAME: %[[AI:.*]]: index
// INDEX64-SAME: %[[AI:.*]]: index
func.func @index_cast_from(%a: index) -> (i64, i32, i64, i32) {
  // INDEX32: %[[A:.*]] = builtin.unrealized_conversion_cast %[[AI]] : index to i32
  // INDEX64: %[[A:.*]] = builtin.unrealized_conversion_cast %[[AI]] : index to i64

  // INDEX32: %[[V0:.*]] = llvm.sext %[[A]] : i32 to i64
  %0 = index.casts %a : index to i64
  // INDEX64: %[[V1:.*]] = llvm.trunc %[[A]] : i64 to i32
  %1 = index.casts %a : index to i32
  // INDEX32: %[[V2:.*]] = llvm.zext %[[A]] : i32 to i64
  %2 = index.castu %a : index to i64
  // INDEX64: %[[V3:.*]] = llvm.trunc %[[A]] : i64 to i32
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
  // INDEX64: %[[V0:.*]] = llvm.sext %[[A]] : i32 to i64
  %0 = index.casts %a : i32 to index
  // INDEX32: %[[V1:.*]] = llvm.trunc %[[B]] : i64 to i32
  %1 = index.casts %b : i64 to index
  // INDEX64: %[[V2:.*]] = llvm.zext %[[A]] : i32 to i64
  %2 = index.castu %a : i32 to index
  // INDEX32: %[[V3:.*]] = llvm.trunc %[[B]] : i64 to i32
  %3 = index.castu %b : i64 to index
  return %0, %1, %2, %3 : index, index, index, index
}

// INDEX32-LABEL: @index_sizeof
// INDEX64-LABEL: @index_sizeof
func.func @index_sizeof() {
  // INDEX32-NEXT: llvm.mlir.constant(32 : i32)
  // INDEX64-NEXT: llvm.mlir.constant(64 : i64)
  %0 = index.sizeof
  return
}

// INDEX32-LABEL: @index_constant
// INDEX64-LABEL: @index_constant
func.func @index_constant() {
  // INDEX32: llvm.mlir.constant(-2100000000 : i32) : i32
  // INDEX64: llvm.mlir.constant(-2100000000 : i64) : i64
  %0 = index.constant -2100000000
  // INDEX32: llvm.mlir.constant(2100000000 : i32) : i32
  // INDEX64: llvm.mlir.constant(2100000000 : i64) : i64
  %1 = index.constant 2100000000
  // INDEX32: llvm.mlir.constant(1294967296 : i32) : i32
  // INDEX64: llvm.mlir.constant(-3000000000 : i64) : i64
  %2 = index.constant -3000000000
  // INDEX32: llvm.mlir.constant(-1294967296 : i32) : i32
  // INDEX64: llvm.mlir.constant(3000000000 : i64) : i64
  %3 = index.constant 3000000000
  return
}
