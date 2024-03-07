// RUN: mlir-opt -convert-arith-to-emitc %s | FileCheck %s

func.func @arith_ops(%arg0: f32, %arg1: f32) {
  // CHECK: [[V0:[^ ]*]] = emitc.add %arg0, %arg1 : (f32, f32) -> f32
  %0 = arith.addf %arg0, %arg1 : f32
  // CHECK: [[V1:[^ ]*]] = emitc.div %arg0, %arg1 : (f32, f32) -> f32
  %1 = arith.divf %arg0, %arg1 : f32  
  // CHECK: [[V2:[^ ]*]] = emitc.mul %arg0, %arg1 : (f32, f32) -> f32
  %2 = arith.mulf %arg0, %arg1 : f32
  // CHECK: [[V3:[^ ]*]] = emitc.sub %arg0, %arg1 : (f32, f32) -> f32
  %3 = arith.subf %arg0, %arg1 : f32

  return
}
