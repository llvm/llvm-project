// RUN: mlir-opt -split-input-file -convert-pdl-to-pdl-interp %s | FileCheck %s

// CHECK:spirv.func @func()
// CHECK-NEXT:%cst0_ui8 = spirv.Constant 0 : ui8
spirv.func @func() -> () "None" { 
  %5 = spirv.Constant 0 : ui8  
  spirv.Return
} 
