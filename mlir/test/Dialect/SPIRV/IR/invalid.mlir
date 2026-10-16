// RUN: mlir-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// spirv.LoadOp
//===----------------------------------------------------------------------===//

func.func @aligned_load_non_positive() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error@below {{'spirv.Load' op attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  %1 = spirv.Load "Function" %0 ["Aligned", 0] : f32
  return
}

// -----

func.func @aligned_load_non_power_of_two() -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error@below {{'spirv.Load' op attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  %1 = spirv.Load "Function" %0 ["Aligned", 3] : f32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.StoreOp
//===----------------------------------------------------------------------===//

func.func @aligned_store_non_positive(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error@below {{'spirv.Store' op attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  spirv.Store "Function" %0, %arg0 ["Aligned", 0] : f32
  return
}

// -----

func.func @aligned_store_non_power_of_two(%arg0 : f32) -> () {
  %0 = spirv.Variable : !spirv.ptr<f32, Function>
  // expected-error@below {{'spirv.Store' op attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  spirv.Store "Function" %0, %arg0 ["Aligned", 3] : f32
  return
}
