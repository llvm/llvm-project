// RUN: mlir-opt %s -inline | FileCheck %s

!qalias = !quant.uniform<i8:f32, 2.0:10>

func.func @inner_dcast_inlinable(%arg0: !qalias) -> f32 {
  %0 = quant.dcast %arg0 : !qalias to f32
  return %0 : f32
}

//  CHECK-LABEL: func.func @test_inline_dcast(
//  CHECK-NOT:    func.call
//  CHECK-NEXT:   quant.dcast
func.func @test_inline_dcast(%v: !qalias) -> f32 {
  %0 = call @inner_dcast_inlinable(%v) : (!qalias) -> f32
  return %0 : f32
}



func.func @inner_qcast_inlinable(%v: f32) -> !qalias {
  %1 = quant.qcast %v : f32 to !qalias
  return %1 : !qalias
}

//  CHECK-LABEL: func.func @test_inline_qcast(
//  CHECK-NOT:    func.call
//  CHECK-NEXT:   quant.qcast
func.func @test_inline_qcast(%v: f32) -> !qalias {
  %0 = call @inner_qcast_inlinable(%v) : (f32) -> !qalias
  return %0 : !qalias
}

func.func @inner_scast_inlinable(%v: i8) -> !qalias {
  %1 = quant.scast %v : i8 to !qalias
  return %1 : !qalias
}

//  CHECK-LABEL: func.func @test_inline_scast(
//  CHECK-NOT:    func.call
//  CHECK-NEXT:   quant.scast
func.func @test_inline_scast(%v: i8) -> !qalias {
  %0 = call @inner_scast_inlinable(%v) : (i8) -> !qalias
  return %0 : !qalias
}



