// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
func.func @unary_ro_e8mf8(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8mf8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e8mf4(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8mf4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e8mf2(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8mf2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e8m1(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8m1 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e8m2(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8m2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e8m4(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8m4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e8m8(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e8m8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// -----
func.func @unary_e8mf8(%rvl: ui32) -> vector<[1] x i8>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[1] x i8>
  return %0 : vector<[1] x i8>
}

func.func @unary_e8mf4(%rvl: ui32) -> vector<[2] x i8>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[2] x i8>
  return %0 : vector<[2] x i8>
}

func.func @unary_e8mf2(%rvl: ui32) -> vector<[4] x i8>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[4] x i8>
  return %0 : vector<[4] x i8>
}

func.func @unary_e8m1(%rvl: ui32) -> vector<[8] x i8>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[8] x i8>
  return %0 : vector<[8] x i8>
}

func.func @unary_e8m2(%rvl: ui32) -> vector<[16] x i8>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[16] x i8>
  return %0 : vector<[16] x i8>
}

func.func @unary_e8m4(%rvl: ui32) -> vector<[32] x i8>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[32] x i8>
  return %0 : vector<[32] x i8>
}

func.func @unary_e8m8(%rvl: ui32) -> vector<[64] x i8>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[64] x i8>
  return %0 : vector<[64] x i8>
}

// -----
func.func @unary_ro_e16mf4(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16mf4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e16mf2(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16mf2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e16m1(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16m1 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e16m2(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16m2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e16m4(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16m4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e16m8(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e16m8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// -----
func.func @unary_e16mf4(%rvl: ui32) -> vector<[1] x i16>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[1] x i16>
  return %0 : vector<[1] x i16>
}

func.func @unary_e16mf2(%rvl: ui32) -> vector<[2] x i16>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[2] x i16>
  return %0 : vector<[2] x i16>
}

func.func @unary_e16m1(%rvl: ui32) -> vector<[4] x i16>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[4] x i16>
  return %0 : vector<[4] x i16>
}

func.func @unary_e16m2(%rvl: ui32) -> vector<[8] x i16>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[8] x i16>
  return %0 : vector<[8] x i16>
}

func.func @unary_e16m4(%rvl: ui32) -> vector<[16] x i16>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[16] x i16>
  return %0 : vector<[16] x i16>
}

func.func @unary_e16m8(%rvl: ui32) -> vector<[32] x i16>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[32] x i16>
  return %0 : vector<[32] x i16>
}

// -----
func.func @unary_ro_e32mf2(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32mf2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e32m1(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32m1 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e32m2(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32m2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e32m4(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32m4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e32m8(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e32m8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// -----
func.func @unary_e32mf2(%rvl: ui32) -> vector<[1] x i32>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[1] x i32>
  return %0 : vector<[1] x i32>
}

func.func @unary_e32m1(%rvl: ui32) -> vector<[2] x i32>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[2] x i32>
  return %0 : vector<[2] x i32>
}

func.func @unary_e32m2(%rvl: ui32) -> vector<[4] x i32>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[4] x i32>
  return %0 : vector<[4] x i32>
}

func.func @unary_e32m4(%rvl: ui32) -> vector<[8] x i32>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[8] x i32>
  return %0 : vector<[8] x i32>
}

func.func @unary_e32m8(%rvl: ui32) -> vector<[16] x i32>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[16] x i32>
  return %0 : vector<[16] x i32>
}

// -----
func.func @unary_ro_e64m1(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e64m1 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e64m2(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e64m2 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e64m4(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e64m4 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

func.func @unary_ro_e64m8(%rvl: ui32) {
  %const = arith.constant 0 : i32
  vcix.unary.ro e64m8 %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5, rd = 30 : i5 } : (i32, ui32)
  return
}

// -----
func.func @unary_e64m1(%rvl: ui32) -> vector<[1] x i64>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[1] x i64>
  return %0 : vector<[1] x i64>
}

func.func @unary_e64m2(%rvl: ui32) -> vector<[2] x i64>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[2] x i64>
  return %0 : vector<[2] x i64>
}

func.func @unary_e64m4(%rvl: ui32) -> vector<[4] x i64>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[4] x i64>
  return %0 : vector<[4] x i64>
}

func.func @unary_e64m8(%rvl: ui32) -> vector<[8] x i64>{
  %const = arith.constant 0 : i32
  %0 = vcix.unary %const, %rvl { opcode = 3 : i2, rs2 = 31 : i5} : (i32, ui32) -> vector<[8] x i64>
  return %0 : vector<[8] x i64>
}

// -----
func.func @binary_vv_ro(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %rvl : ui32) {
  vcix.binary.ro %op1, %op2, %rvl { opcode = 3 : i2, rd = 30 : i5 } : (vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

func.func @binary_vv(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

func.func @binary_xv_ro(%op1: i32, %op2 : vector<[4] x f32>, %rvl : ui32) {
  vcix.binary.ro %op1, %op2, %rvl { opcode = 3 : i2, rd = 30 : i5 } : (i32, vector<[4] x f32>, ui32)
  return
}

func.func @binary_xv(%op1: i32, %op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

func.func @binary_fv_ro(%op1: f32, %op2 : vector<[4] x f32>, %rvl : ui32) {
  vcix.binary.ro %op1, %op2, %rvl { opcode = 1 : i1, rd = 30 : i5 } : (f32, vector<[4] x f32>, ui32)
  return
}

func.func @binary_fv(%op1: f32, %op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  %0 = vcix.binary %op1, %op2, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

func.func @binary_iv_ro(%op2 : vector<[4] x f32>, %rvl : ui32) {
  %const = arith.constant 1 : i5
  vcix.binary.ro %const, %op2, %rvl { opcode = 3 : i2, rd = 30 : i5 } : (i5, vector<[4] x f32>, ui32)
  return
}

func.func @binary_iv(%op2 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  %const = arith.constant 1 : i5
  %0 = vcix.binary %const, %op2, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// -----
func.func @ternary_vvv_ro(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) {
  vcix.ternary.ro %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

func.func @ternary_vvv(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  %0 = vcix.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

func.func @ternary_xvv_ro(%op1: i32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) {
  vcix.ternary.ro %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

func.func @ternary_xvv(%op1: i32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  %0 = vcix.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

func.func @ternary_fvv_ro(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) {
  vcix.ternary.ro %op1, %op2, %op3, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

func.func @ternary_fvv(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  %0 = vcix.ternary %op1, %op2, %op3, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

func.func @ternary_ivv_ro(%op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) {
  %const = arith.constant 1 : i5
  vcix.ternary.ro %const, %op2, %op3, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, vector<[4] x f32>, ui32)
  return
}

func.func @ternary_ivv(%op2 : vector<[4] x f32>, %op3 : vector<[4] x f32>, %rvl : ui32) -> vector<[4] x f32> {
  %const = arith.constant 1 : i5
  %0 = vcix.ternary %const, %op2, %op3, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, vector<[4] x f32>, ui32) -> vector<[4] x f32>
  return %0 : vector<[4] x f32>
}

// -----
func.func @wide_ternary_vvw_ro(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) {
  vcix.wide.ternary.ro %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f64>, ui32)
  return
}

func.func @wide_ternary_vvw(%op1: vector<[4] x f32>, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> {
  %0 = vcix.wide.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (vector<[4] x f32>, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %0: vector<[4] x f64>
}

func.func @wide_ternary_xvw_ro(%op1: i32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) {
  vcix.wide.ternary.ro %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, vector<[4] x f64>, ui32)
  return
}

func.func @wide_ternary_xvw(%op1: i32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> {
  %0 = vcix.wide.ternary %op1, %op2, %op3, %rvl { opcode = 3 : i2 } : (i32, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %0 : vector<[4] x f64>
}

func.func @wide_ternary_fvw_ro(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) {
  vcix.wide.ternary.ro %op1, %op2, %op3, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, vector<[4] x f64>, ui32)
  return
}

func.func @wide_ternary_fvw(%op1: f32, %op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> {
  %0 = vcix.wide.ternary %op1, %op2, %op3, %rvl { opcode = 1 : i1 } : (f32, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %op3 : vector<[4] x f64>
}

func.func @wide_ternary_ivw_ro(%op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) {
  %const = arith.constant 1 : i5
  vcix.wide.ternary.ro %const, %op2, %op3, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, vector<[4] x f64>, ui32)
  return
}

func.func @wide_ternary_ivv(%op2 : vector<[4] x f32>, %op3 : vector<[4] x f64>, %rvl : ui32) -> vector<[4] x f64> {
  %const = arith.constant 1 : i5
  %0 = vcix.wide.ternary %const, %op2, %op3, %rvl { opcode = 3 : i2 } : (i5, vector<[4] x f32>, vector<[4] x f64>, ui32) -> vector<[4] x f64>
  return %op3 : vector<[4] x f64>
}

// -----
func.func @fixed_binary_vv_ro(%op1: vector<4 x f32>, %op2 : vector<4 x f32>) {
  vcix.binary.ro %op1, %op2 { opcode = 3 : i2, rd = 30 : i5 } : (vector<4 x f32>, vector<4 x f32>)
  return
}

func.func @fixed_binary_vv(%op1: vector<4 x f32>, %op2 : vector<4 x f32>) -> vector<4 x f32> {
  %0 = vcix.binary %op1, %op2 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

func.func @fixed_binary_xv_ro(%op1: i32, %op2 : vector<4 x f32>) {
  vcix.binary.ro %op1, %op2 { opcode = 3 : i2, rd = 30 : i5 } : (i32, vector<4 x f32>)
  return
}

func.func @fixed_binary_xv(%op1: i32, %op2 : vector<4 x f32>) -> vector<4 x f32> {
  %0 = vcix.binary %op1, %op2 { opcode = 3 : i2 } : (i32, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

func.func @fixed_binary_fv_ro(%op1: f32, %op2 : vector<4 x f32>) {
  vcix.binary.ro %op1, %op2 { opcode = 1 : i1, rd = 30 : i5 } : (f32, vector<4 x f32>)
  return
}

func.func @fixed_binary_fv(%op1: f32, %op2 : vector<4 x f32>) -> vector<4 x f32> {
  %0 = vcix.binary %op1, %op2 { opcode = 1 : i1 } : (f32, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

func.func @fixed_binary_iv_ro(%op2 : vector<4 x f32>) {
  %const = arith.constant 1 : i5
  vcix.binary.ro %const, %op2 { opcode = 3 : i2, rd = 30 : i5 } : (i5, vector<4 x f32>)
  return
}

func.func @fixed_binary_iv(%op2 : vector<4 x f32>) -> vector<4 x f32> {
  %const = arith.constant 1 : i5
  %0 = vcix.binary %const, %op2 { opcode = 3 : i2 } : (i5, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// -----
func.func @fixed_ternary_vvv_ro(%op1: vector<4 x f32>, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) {
  vcix.ternary.ro %op1, %op2, %op3 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>, vector<4 x f32>)
  return
}

func.func @fixed_ternary_vvv(%op1: vector<4 x f32>, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) -> vector<4 x f32> {
  %0 = vcix.ternary %op1, %op2, %op3 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

func.func @fixed_ternary_xvv_ro(%op1: i32, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) {
  vcix.ternary.ro %op1, %op2, %op3 { opcode = 3 : i2 } : (i32, vector<4 x f32>, vector<4 x f32>)
  return
}

func.func @fixed_ternary_xvv(%op1: i32, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) -> vector<4 x f32> {
  %0 = vcix.ternary %op1, %op2, %op3 { opcode = 3 : i2 } : (i32, vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

func.func @fixed_ternary_fvv_ro(%op1: f32, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) {
  vcix.ternary.ro %op1, %op2, %op3 { opcode = 1 : i1 } : (f32, vector<4 x f32>, vector<4 x f32>)
  return
}

func.func @fixed_ternary_fvv(%op1: f32, %op2 : vector<4 x f32>, %op3 : vector<4 x f32>) -> vector<4 x f32> {
  %0 = vcix.ternary %op1, %op2, %op3 { opcode = 1 : i1 } : (f32, vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

func.func @fixed_ternary_ivv_ro(%op2 : vector<4 x f32>, %op3 : vector<4 x f32>) {
  %const = arith.constant 1 : i5
  vcix.ternary.ro %const, %op2, %op3 { opcode = 3 : i2 } : (i5, vector<4 x f32>, vector<4 x f32>)
  return
}

func.func @fixed_ternary_ivv(%op2 : vector<4 x f32>, %op3 : vector<4 x f32>) -> vector<4 x f32> {
  %const = arith.constant 1 : i5
  %0 = vcix.ternary %const, %op2, %op3 { opcode = 3 : i2 } : (i5, vector<4 x f32>, vector<4 x f32>) -> vector<4 x f32>
  return %0 : vector<4 x f32>
}

// -----
func.func @fixed_wide_ternary_vvw_ro(%op1: vector<4 x f32>, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) {
  vcix.wide.ternary.ro %op1, %op2, %op3 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>, vector<4 x f64>)
  return
}

func.func @fixed_wide_ternary_vvw(%op1: vector<4 x f32>, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) -> vector<4 x f64> {
  %0 = vcix.wide.ternary %op1, %op2, %op3 { opcode = 3 : i2 } : (vector<4 x f32>, vector<4 x f32>, vector<4 x f64>) -> vector<4 x f64>
  return %0 : vector<4 x f64>
}

func.func @fixed_wide_ternary_xvw_ro(%op1: i32, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) {
  vcix.wide.ternary.ro %op1, %op2, %op3 { opcode = 3 : i2 } : (i32, vector<4 x f32>, vector<4 x f64>)
  return
}

func.func @fixed_wide_ternary_xvw(%op1: i32, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) -> vector<4 x f64> {
  %0 = vcix.wide.ternary %op1, %op2, %op3 { opcode = 3 : i2 } : (i32, vector<4 x f32>, vector<4 x f64>) -> vector<4 x f64>
  return %0 : vector<4 x f64>
}

func.func @fixed_wide_ternary_fvw_ro(%op1: f32, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) {
  vcix.wide.ternary.ro %op1, %op2, %op3 { opcode = 1 : i1 } : (f32, vector<4 x f32>, vector<4 x f64>)
  return
}

func.func @fixed_wide_ternary_fvw(%op1: f32, %op2 : vector<4 x f32>, %op3 : vector<4 x f64>) -> vector<4 x f64> {
  %0 = vcix.wide.ternary %op1, %op2, %op3 { opcode = 1 : i1 } : (f32, vector<4 x f32>, vector<4 x f64>) -> vector<4 x f64>
  return %op3 : vector<4 x f64>
}

func.func @fixed_wide_ternary_ivw_ro(%op2 : vector<4 x f32>, %op3 : vector<4 x f64>) {
  %const = arith.constant 1 : i5
  vcix.wide.ternary.ro %const, %op2, %op3 { opcode = 3 : i2 } : (i5, vector<4 x f32>, vector<4 x f64>)
  return
}

func.func @fixed_wide_ternary_ivv(%op2 : vector<4 x f32>, %op3 : vector<4 x f64>) -> vector<4 x f64> {
  %const = arith.constant 1 : i5
  %0 = vcix.wide.ternary %const, %op2, %op3 { opcode = 3 : i2 } : (i5, vector<4 x f32>, vector<4 x f64>) -> vector<4 x f64>
  return %op3 : vector<4 x f64>
}
