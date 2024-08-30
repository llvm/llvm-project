// RUN: mlir-opt %s -llvm-optimize-for-nvvm-target | FileCheck %s

// CHECK-LABEL: llvm.func @fdiv_fp16
llvm.func @fdiv_fp16(%arg0 : f16, %arg1 : f16) -> f16 {
  // CHECK-DAG: %[[c0:.*]]      = llvm.mlir.constant(0 : ui32) : i32
  // CHECK-DAG: %[[mask:.*]]    = llvm.mlir.constant(2139095040 : ui32) : i32
  // CHECK-DAG: %[[lhs:.*]]     = llvm.fpext %arg0 : f16 to f32
  // CHECK-DAG: %[[rhs:.*]]     = llvm.fpext %arg1 : f16 to f32
  // CHECK-DAG: %[[rcp:.*]]     = nvvm.rcp.approx.ftz.f %[[rhs]] : f32
  // CHECK-DAG: %[[approx:.*]]  = llvm.fmul %[[lhs]], %[[rcp]] : f32
  // CHECK-DAG: %[[neg:.*]]     = llvm.fneg %[[rhs]] : f32
  // CHECK-DAG: %[[err:.*]]     = llvm.intr.fma(%[[approx]], %[[neg]], %[[lhs]]) : (f32, f32, f32) -> f32
  // CHECK-DAG: %[[refined:.*]] = llvm.intr.fma(%[[err]], %[[rcp]], %[[approx]]) : (f32, f32, f32) -> f32
  // CHECK-DAG: %[[cast:.*]]    = llvm.bitcast %[[approx]] : f32 to i32
  // CHECK-DAG: %[[exp:.*]]     = llvm.and %[[cast]], %[[mask]] : i32
  // CHECK-DAG: %[[is_zero:.*]] = llvm.icmp "eq" %[[exp]], %[[c0]] : i32
  // CHECK-DAG: %[[is_mask:.*]] = llvm.icmp "eq" %[[exp]], %[[mask]] : i32
  // CHECK-DAG: %[[pred:.*]]    = llvm.or %[[is_zero]], %[[is_mask]] : i1
  // CHECK-DAG: %[[select:.*]]  = llvm.select %[[pred]], %[[approx]], %[[refined]] : i1, f32
  // CHECK-DAG: %[[result:.*]]  = llvm.fptrunc %[[select]] : f32 to f16
  %result = llvm.fdiv %arg0, %arg1 : f16
  // CHECK: llvm.return %[[result]] : f16
  llvm.return %result : f16
}

// CHECK-LABEL: llvm.func @ui16_to_f32
llvm.func @ui16_to_f32(%arg0 : i16) -> f32 {
  // CHECK-DAG: %[[zext:.*]] = llvm.zext %arg0 : i16 to i32
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(1262485504 : i32) : i32
  // CHECK-DAG: %[[add:.*]] = llvm.add %[[zext]], %[[bias]] : i32
  // CHECK-DAG: %[[cast:.*]] = llvm.bitcast %[[add]] : i32 to f32
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(0x4B400000 : f32) : f32
  // CHECK-DAG: %[[result:.*]] = llvm.fsub %[[cast]], %[[bias]]  : f32
  %result = llvm.uitofp %arg0 : i16 to f32
  // CHECK: llvm.return %[[result]] : f32
  llvm.return %result : f32
}

// Checks that expansion only applies to integer width up to mantissa width.
// CHECK-LABEL: llvm.func @si32_to_float
llvm.func @si32_to_float_no_rewrite(%arg0 : i32) -> f32 {
  // CHECK: %[[result:.*]] = llvm.sitofp %arg0 : i32 to f32
  %result = llvm.sitofp %arg0 : i32 to f32
  // CHECK: llvm.return %[[result]] : f32
  llvm.return %result : f32
}

// CHECK-LABEL: llvm.func @si8_to_f16
llvm.func @si8_to_f16(%arg0 : i8) -> f16 {
  // CHECK-DAG: %[[sext:.*]] = llvm.sext %arg0 : i8 to i16
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(26112 : i16) : i16
  // CHECK-DAG: %[[add:.*]] = llvm.add %[[sext]], %[[bias]] : i16
  // CHECK-DAG: %[[cast:.*]] = llvm.bitcast %[[add]] : i16 to f16
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(1.536000e+03 : f16) : f16
  // CHECK-DAG: %[[result:.*]] = llvm.fsub %[[cast]], %[[bias]]  : f16
  %result = llvm.sitofp %arg0 : i8 to f16
  // CHECK: llvm.return %[[result]] : f16
  llvm.return %result : f16
}

// CHECK-LABEL: llvm.func @vec_ui4_to_bf16
llvm.func @vec_ui4_to_bf16(%arg0 : vector<4xi4>) -> vector<4xbf16> {
  // CHECK-DAG: %[[zext:.*]] = llvm.zext %arg0 : vector<4xi4> to vector<4xi16>
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(dense<17216> : vector<4xi16>) : vector<4xi16>
  // CHECK-DAG: %[[add:.*]] = llvm.add %[[zext]], %[[bias]] : vector<4xi16>
  // CHECK-DAG: %[[cast:.*]] = llvm.bitcast %[[add]] : vector<4xi16> to vector<4xbf16>
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(dense<1.920000e+02> : vector<4xbf16>) : vector<4xbf16>
  // CHECK-DAG: %[[result:.*]] = llvm.fsub %[[cast]], %[[bias]]  : vector<4xbf16>
  %result = llvm.uitofp %arg0 : vector<4xi4> to vector<4xbf16>
  // CHECK: llvm.return %[[result]] : vector<4xbf16>
  llvm.return %result : vector<4xbf16>
}

// Checks code path when integer width is equal to mantissa width.
// CHECK-LABEL: llvm.func @vec_si8_to_bf16
llvm.func @vec_si8_to_bf16(%arg0 : vector<4xi8>) -> vector<4xbf16> {
  // CHECK-DAG: %[[zext:.*]] = llvm.zext %arg0 : vector<4xi8> to vector<4xi16>
  // CHECK-DAG: %[[const:.*]] = llvm.mlir.constant(dense<17152> : vector<4xi16>) : vector<4xi16>
  // CHECK-DAG: %[[or:.*]] = llvm.or %[[zext]], %[[const]] : vector<4xi16>
  // CHECK-DAG: %[[exp_mask:.*]] = llvm.mlir.constant(dense<-128> : vector<4xi16>) : vector<4xi16>
  // CHECK-DAG: %[[man_mask:.*]] = llvm.mlir.constant(dense<-129> : vector<4xi16>) : vector<4xi16>
  // CHECK-DAG: %[[exp_and:.*]] = llvm.and %[[or]], %[[exp_mask]]  : vector<4xi16>
  // CHECK-DAG: %[[man_and:.*]] = llvm.and %[[or]], %[[man_mask]]  : vector<4xi16>
  // CHECK-DAG: %[[exp_cast:.*]] = llvm.bitcast %[[exp_and]] : vector<4xi16> to vector<4xbf16>
  // CHECK-DAG: %[[man_cast:.*]] = llvm.bitcast %[[man_and]] : vector<4xi16> to vector<4xbf16>
  // CHECK-DAG: %[[result:.*]] = llvm.fsub %[[man_cast]], %[[exp_cast]]  : vector<4xbf16>
  %result = llvm.sitofp %arg0 : vector<4xi8> to vector<4xbf16>
  // CHECK: llvm.return %[[result]] : vector<4xbf16>
  llvm.return %result : vector<4xbf16>
}

// Checks code path when integer width is equal to mantissa width.
// CHECK-LABEL: llvm.func @ui8_to_bf16
llvm.func @ui8_to_bf16(%arg0 : i8) -> bf16 {
  // CHECK-DAG: %[[zext:.*]] = llvm.zext %arg0 : i8 to i16
  // CHECK-DAG: %[[const:.*]] = llvm.mlir.constant(17152 : i16) : i16
  // CHECK-DAG: %[[or:.*]] = llvm.or %[[zext]], %[[const]] : i16
  // CHECK-DAG: %[[exp_mask:.*]] = llvm.mlir.constant(-128 : i16) : i16
  // CHECK-DAG: %[[man_mask:.*]] = llvm.mlir.constant(-129 : i16) : i16
  // CHECK-DAG: %[[exp_and:.*]] = llvm.and %[[or]], %[[exp_mask]]  : i16
  // CHECK-DAG: %[[man_and:.*]] = llvm.and %[[or]], %[[man_mask]]  : i16
  // CHECK-DAG: %[[exp_cast:.*]] = llvm.bitcast %[[exp_and]] : i16 to bf16
  // CHECK-DAG: %[[man_cast:.*]] = llvm.bitcast %[[man_and]] : i16 to bf16
  // CHECK-DAG: %[[result:.*]] = llvm.fadd %[[man_cast]], %[[exp_cast]]  : bf16
  %result = llvm.uitofp %arg0 : i8 to bf16
  // CHECK: llvm.return %[[result]] : bf16
  llvm.return %result : bf16
}

// Checks that expansion does not apply when exponent bias lsb is set.
// CHECK-LABEL: llvm.func @ui11_to_f16
llvm.func @ui11_to_f16(%arg0 : i11) -> f16 {
  // CHECK: %[[result:.*]] = llvm.uitofp %arg0 : i11 to f16
  %result = llvm.uitofp %arg0 : i11 to f16
  // CHECK: llvm.return %[[result]] : f16
  llvm.return %result : f16
}

// CHECK-LABEL: llvm.func @ui16_to_f32
llvm.func @ui16_to_f32(%arg0 : i16) -> f32 {
  // CHECK-DAG: %[[zext:.*]] = llvm.zext %arg0 : i16 to i32
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(1262485504 : i32) : i32
  // CHECK-DAG: %[[add:.*]] = llvm.add %[[zext]], %[[bias]] : i32
  // CHECK-DAG: %[[cast:.*]] = llvm.bitcast %[[add]] : i32 to f32
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(0x4B400000 : f32) : f32
  // CHECK-DAG: %[[result:.*]] = llvm.fsub %[[cast]], %[[bias]]  : f32
  %result = llvm.uitofp %arg0 : i16 to f32
  // CHECK: llvm.return %[[result]] : f32
  llvm.return %result : f32
}

// Checks that expansion only applies to integer width up to mantissa width.
// CHECK-LABEL: llvm.func @si32_to_float
llvm.func @si32_to_float_no_rewrite(%arg0 : i32) -> f32 {
  // CHECK: %[[result:.*]] = llvm.sitofp %arg0 : i32 to f32
  %result = llvm.sitofp %arg0 : i32 to f32
  // CHECK: llvm.return %[[result]] : f32
  llvm.return %result : f32
}

// CHECK-LABEL: llvm.func @si8_to_f16
llvm.func @si8_to_f16(%arg0 : i8) -> f16 {
  // CHECK-DAG: %[[sext:.*]] = llvm.sext %arg0 : i8 to i16
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(26112 : i16) : i16
  // CHECK-DAG: %[[add:.*]] = llvm.add %[[sext]], %[[bias]] : i16
  // CHECK-DAG: %[[cast:.*]] = llvm.bitcast %[[add]] : i16 to f16
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(1.536000e+03 : f16) : f16
  // CHECK-DAG: %[[result:.*]] = llvm.fsub %[[cast]], %[[bias]]  : f16
  %result = llvm.sitofp %arg0 : i8 to f16
  // CHECK: llvm.return %[[result]] : f16
  llvm.return %result : f16
}

// CHECK-LABEL: llvm.func @vec_ui4_to_bf16
llvm.func @vec_ui4_to_bf16(%arg0 : vector<4xi4>) -> vector<4xbf16> {
  // CHECK-DAG: %[[zext:.*]] = llvm.zext %arg0 : vector<4xi4> to vector<4xi16>
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(dense<17216> : vector<4xi16>) : vector<4xi16>
  // CHECK-DAG: %[[add:.*]] = llvm.add %[[zext]], %[[bias]] : vector<4xi16>
  // CHECK-DAG: %[[cast:.*]] = llvm.bitcast %[[add]] : vector<4xi16> to vector<4xbf16>
  // CHECK-DAG: %[[bias:.*]] = llvm.mlir.constant(dense<1.920000e+02> : vector<4xbf16>) : vector<4xbf16>
  // CHECK-DAG: %[[result:.*]] = llvm.fsub %[[cast]], %[[bias]]  : vector<4xbf16>
  %result = llvm.uitofp %arg0 : vector<4xi4> to vector<4xbf16>
  // CHECK: llvm.return %[[result]] : vector<4xbf16>
  llvm.return %result : vector<4xbf16>
}

// Checks code path when integer width is equal to mantissa width.
// CHECK-LABEL: llvm.func @vec_si8_to_bf16
llvm.func @vec_si8_to_bf16(%arg0 : vector<4xi8>) -> vector<4xbf16> {
  // CHECK-DAG: %[[zext:.*]] = llvm.zext %arg0 : vector<4xi8> to vector<4xi16>
  // CHECK-DAG: %[[const:.*]] = llvm.mlir.constant(dense<17152> : vector<4xi16>) : vector<4xi16>
  // CHECK-DAG: %[[or:.*]] = llvm.or %[[zext]], %[[const]] : vector<4xi16>
  // CHECK-DAG: %[[exp_mask:.*]] = llvm.mlir.constant(dense<-128> : vector<4xi16>) : vector<4xi16>
  // CHECK-DAG: %[[man_mask:.*]] = llvm.mlir.constant(dense<-129> : vector<4xi16>) : vector<4xi16>
  // CHECK-DAG: %[[exp_and:.*]] = llvm.and %[[or]], %[[exp_mask]]  : vector<4xi16>
  // CHECK-DAG: %[[man_and:.*]] = llvm.and %[[or]], %[[man_mask]]  : vector<4xi16>
  // CHECK-DAG: %[[exp_cast:.*]] = llvm.bitcast %[[exp_and]] : vector<4xi16> to vector<4xbf16>
  // CHECK-DAG: %[[man_cast:.*]] = llvm.bitcast %[[man_and]] : vector<4xi16> to vector<4xbf16>
  // CHECK-DAG: %[[result:.*]] = llvm.fsub %[[man_cast]], %[[exp_cast]]  : vector<4xbf16>
  %result = llvm.sitofp %arg0 : vector<4xi8> to vector<4xbf16>
  // CHECK: llvm.return %[[result]] : vector<4xbf16>
  llvm.return %result : vector<4xbf16>
}

// Checks that expansion does not apply when unsigned integer width is equal to
// mantissa width.
// CHECK-LABEL: llvm.func @ui8_to_bf16
llvm.func @ui8_to_bf16(%arg0 : i8) -> bf16 {
  // CHECK: %[[result:.*]] = llvm.uitofp %arg0 : i8 to bf16
  %result = llvm.uitofp %arg0 : i8 to bf16
  // CHECK: llvm.return %[[result]] : bf16
  llvm.return %result : bf16
}

// Checks that expansion does not apply when exponent bias lsb is set.
// CHECK-LABEL: llvm.func @ui11_to_f16
llvm.func @ui11_to_f16(%arg0 : i11) -> f16 {
  // CHECK: %[[result:.*]] = llvm.uitofp %arg0 : i11 to f16
  %result = llvm.uitofp %arg0 : i11 to f16
  // CHECK: llvm.return %[[result]] : f16
  llvm.return %result : f16
}
