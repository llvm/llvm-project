// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.EmitVertex
//===----------------------------------------------------------------------===//

func.func @emit_vertex() {
  // CHECK: spirv.EmitVertex
  spirv.EmitVertex
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.EndPrimitive
//===----------------------------------------------------------------------===//

func.func @end_primitive() {
  // CHECK: spirv.EndPrimitive
  spirv.EndPrimitive
  spirv.Return
}
