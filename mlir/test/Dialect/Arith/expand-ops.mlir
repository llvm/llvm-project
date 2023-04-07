// RUN: mlir-opt %s -arith-expand="include-bf16=true" --canonicalize -split-input-file | FileCheck %s

func.func @truncf_f32(%arg0 : f32) -> bf16 {
    %0 = arith.truncf %arg0 : f32 to bf16
    return %0 : bf16
}

// CHECK-LABEL: @truncf_f32

// CHECK: %[[C16:.+]] = arith.constant 16
// CHECK: %[[C32768:.+]] = arith.constant 32768
// CHECK: %[[C2130706432:.+]] = arith.constant 2130706432
// CHECK: %[[C2139095040:.+]] = arith.constant 2139095040
// CHECK: %[[C8388607:.+]] = arith.constant 8388607
// CHECK: %[[C31:.+]] = arith.constant 31
// CHECK: %[[C23:.+]] = arith.constant 23
// CHECK: %[[BITCAST:.+]] = arith.bitcast %arg0
// CHECK: %[[SIGN:.+]] = arith.shrui %[[BITCAST:.+]], %[[C31]]
// CHECK: %[[ROUND:.+]] = arith.subi %[[C32768]], %[[SIGN]]
// CHECK: %[[MANTISSA:.+]] = arith.andi %[[BITCAST]], %[[C8388607]]
// CHECK: %[[ROUNDED:.+]] = arith.addi %[[MANTISSA]], %[[ROUND]]
// CHECK: %[[ROLL:.+]] = arith.shrui %[[ROUNDED]], %[[C23]]
// CHECK: %[[SHR:.+]] = arith.shrui %[[ROUNDED]], %[[ROLL]]
// CHECK: %[[EXP:.+]] = arith.andi %0, %[[C2139095040]]
// CHECK: %[[EXPROUND:.+]] = arith.addi %[[EXP]], %[[ROUNDED]]
// CHECK: %[[EXPROLL:.+]] = arith.andi %[[EXPROUND]], %[[C2139095040]]
// CHECK: %[[EXPMAX:.+]] = arith.cmpi uge, %[[EXP]], %[[C2130706432]]
// CHECK: %[[EXPNEW:.+]] = arith.select %[[EXPMAX]], %[[EXP]], %[[EXPROLL]]
// CHECK: %[[OVERFLOW_B:.+]] = arith.trunci %[[ROLL]]
// CHECK: %[[KEEP_MAN:.+]] = arith.andi %[[EXPMAX]], %[[OVERFLOW_B]]
// CHECK: %[[MANNEW:.+]] = arith.select %[[KEEP_MAN]], %[[MANTISSA]], %[[SHR]]
// CHECK: %[[NEWSIGN:.+]] = arith.shli %[[SIGN]], %[[C31]]
// CHECK: %[[WITHEXP:.+]] = arith.ori %[[NEWSIGN]], %[[EXPNEW]]
// CHECK: %[[WITHMAN:.+]] = arith.ori %[[WITHEXP]], %[[MANNEW]]
// CHECK: %[[SHIFT:.+]] = arith.shrui %[[WITHMAN]], %[[C16]]
// CHECK: %[[TRUNC:.+]] = arith.trunci %[[SHIFT]]
// CHECK: %[[RES:.+]] = arith.bitcast %[[TRUNC]]
// CHECK: return %[[RES]]

// -----

func.func @truncf_vector_f32(%arg0 : vector<4xf32>) -> vector<4xbf16> {
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xbf16>
    return %0 : vector<4xbf16>
}

// CHECK-LABEL: @truncf_vector_f32
// CHECK-NOT: arith.truncf
