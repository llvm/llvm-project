// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.EmitVertex
//===----------------------------------------------------------------------===//

func.func @emit_vertex() {
  // CHECK: spirv.EmitVertex
  spirv.EmitVertex
  return
}

//===----------------------------------------------------------------------===//
// spirv.EndPrimitive
//===----------------------------------------------------------------------===//

func.func @end_primitive() {
  // CHECK: spirv.EndPrimitive
  spirv.EndPrimitive
  return
}
