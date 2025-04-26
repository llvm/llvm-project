// RUN: mlir-opt %s -test-scf-if-callback-builder -split-input-file | FileCheck %s

// CHECK-LABEL: @callback_break_then
// CHECK: scf.if %{{.*}} -> (i32) {
// CHECK: scf.break
// CHECK: } else {
// CHECK: scf.yield %{{.*}} : i32
func.func @callback_break_then(%cond: i1, %x: i32) -> i32 {
  %r = scf.loop token(%token) -> i32 {
    %v = scf.if %cond -> i32 {
      scf.break [%token] %x : i32
    } else {
      scf.yield %x : i32
    }
    scf.break [%token] %v : i32
  }
  return %r : i32
}

// -----

// CHECK-LABEL: @callback_continue_else
// CHECK: scf.if %{{.*}} -> (i32) {
// CHECK: scf.yield %{{.*}} : i32
// CHECK: } else {
// CHECK: scf.continue
func.func @callback_continue_else(%cond: i1, %x: i32) -> i32 {
  %r = scf.loop token(%token) iter_args(%arg = %x) : i32 -> i32 {
    %v = scf.if %cond -> i32 {
      scf.yield %arg : i32
    } else {
      scf.continue [%token] %arg : i32
    }
    scf.break [%token] %v : i32
  }
  return %r : i32
}

// -----

// CHECK-LABEL: @callback_break_without_else
// CHECK: scf.if %{{.*}} {
// CHECK: scf.break
func.func @callback_break_without_else(%cond: i1) {
  scf.loop token(%token) {
    scf.if %cond {
      scf.break [%token]
    }
  }
  return
}
