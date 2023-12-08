// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

// Deduplicate 2 global variables with the same descriptor set and binding.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.GlobalVariable @foo

// CHECK-NEXT:     spirv.func @use_foo
// CHECK-NEXT:       spirv.mlir.addressof @foo
// CHECK-NEXT:       spirv.Load
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @use_bar
// CHECK-NEXT:       spirv.mlir.addressof @foo
// CHECK-NEXT:       spirv.Load
// CHECK-NEXT:       spirv.FAdd
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>

  spirv.func @use_foo() -> f32 "None" {
    %0 = spirv.mlir.addressof @foo : !spirv.ptr<f32, Input>
    %1 = spirv.Load "Input" %0 : f32
    spirv.ReturnValue %1 : f32
  }
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @bar bind(1, 0) : !spirv.ptr<f32, Input>

  spirv.func @use_bar() -> f32 "None" {
    %0 = spirv.mlir.addressof @bar : !spirv.ptr<f32, Input>
    %1 = spirv.Load "Input" %0 : f32
    %2 = spirv.FAdd %1, %1 : f32
    spirv.ReturnValue %2 : f32
  }
}

// -----

// Deduplicate 2 global variables with the same descriptor set and binding but different types.

// CHECK:      module {
// CHECK-NEXT: spirv.module Logical GLSL450 {
// CHECK-NEXT:   spirv.GlobalVariable @foo bind(1, 0)

// CHECK-NEXT:   spirv.GlobalVariable @bar bind(1, 0)

// CHECK-NEXT:   spirv.func @use_bar
// CHECK-NEXT:     spirv.mlir.addressof @bar
// CHECK-NEXT:     spirv.Load
// CHECK-NEXT:     spirv.ReturnValue
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: }

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<i32, Input>
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @bar bind(1, 0) : !spirv.ptr<f32, Input>

  spirv.func @use_bar() -> f32 "None" {
    %0 = spirv.mlir.addressof @bar : !spirv.ptr<f32, Input>
    %1 = spirv.Load "Input" %0 : f32
    spirv.ReturnValue %1 : f32
  }
}

// -----

// Deduplicate 2 global variables with the same built-in attribute.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.GlobalVariable @foo built_in("GlobalInvocationId")
// CHECK-NEXT:     spirv.func @use_bar
// CHECK-NEXT:       spirv.mlir.addressof @foo
// CHECK-NEXT:       spirv.Load
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @bar built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>

  spirv.func @use_bar() -> vector<3xi32> "None" {
    %0 = spirv.mlir.addressof @bar : !spirv.ptr<vector<3xi32>, Input>
    %1 = spirv.Load "Input" %0 : vector<3xi32>
    spirv.ReturnValue %1 : vector<3xi32>
  }
}

// -----

// Deduplicate 2 spec constants with the same spec ID.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.SpecConstant @foo spec_id(5)

// CHECK-NEXT:     spirv.func @use_foo()
// CHECK-NEXT:       %0 = spirv.mlir.referenceof @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @use_bar()
// CHECK-NEXT:       %0 = spirv.mlir.referenceof @foo
// CHECK-NEXT:       spirv.FAdd
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

spirv.module Logical GLSL450 {
  spirv.SpecConstant @foo spec_id(5) = 1. : f32

  spirv.func @use_foo() -> (f32) "None" {
    %0 = spirv.mlir.referenceof @foo : f32
    spirv.ReturnValue %0 : f32
  }
}

spirv.module Logical GLSL450 {
  spirv.SpecConstant @bar spec_id(5) = 1. : f32

  spirv.func @use_bar() -> (f32) "None" {
    %0 = spirv.mlir.referenceof @bar : f32
    %1 = spirv.FAdd %0, %0 : f32
    spirv.ReturnValue %1 : f32
  }
}

// -----

// Don't deduplicate functions with similar ops but different operands.

//       CHECK: spirv.module Logical GLSL450 {
//  CHECK-NEXT:   spirv.func @foo(%[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32)
//  CHECK-NEXT:     %[[ADD:.+]] = spirv.FAdd %[[ARG0]], %[[ARG1]] : f32
//  CHECK-NEXT:     %[[MUL:.+]] = spirv.FMul %[[ADD]], %[[ARG2]] : f32
//  CHECK-NEXT:     spirv.ReturnValue %[[MUL]] : f32
//  CHECK-NEXT:   }
//  CHECK-NEXT:   spirv.func @foo_1(%[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32)
//  CHECK-NEXT:     %[[ADD:.+]] = spirv.FAdd %[[ARG0]], %[[ARG2]] : f32
//  CHECK-NEXT:     %[[MUL:.+]] = spirv.FMul %[[ADD]], %[[ARG1]] : f32
//  CHECK-NEXT:     spirv.ReturnValue %[[MUL]] : f32
//  CHECK-NEXT:   }
//  CHECK-NEXT: }

spirv.module Logical GLSL450 {
  spirv.func @foo(%a: f32, %b: f32, %c: f32) -> f32 "None" {
    %add = spirv.FAdd %a, %b: f32
    %mul = spirv.FMul %add, %c: f32
    spirv.ReturnValue %mul: f32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%a: f32, %b: f32, %c: f32) -> f32 "None" {
    %add = spirv.FAdd %a, %c: f32
    %mul = spirv.FMul %add, %b: f32
    spirv.ReturnValue %mul: f32
  }
}

// -----

// TODO: re-enable this test once we have better function deduplication.

// XXXXX:      module {
// XXXXX-NEXT:   spirv.module Logical GLSL450 {
// XXXXX-NEXT:     spirv.SpecConstant @bar spec_id(5)

// XXXXX-NEXT:     spirv.func @foo(%arg0: f32)
// XXXXX-NEXT:       spirv.ReturnValue
// XXXXX-NEXT:     }

// XXXXX-NEXT:     spirv.func @foo_different_body(%arg0: f32)
// XXXXX-NEXT:       spirv.mlir.referenceof
// XXXXX-NEXT:       spirv.ReturnValue
// XXXXX-NEXT:     }

// XXXXX-NEXT:     spirv.func @baz(%arg0: i32)
// XXXXX-NEXT:       spirv.ReturnValue
// XXXXX-NEXT:     }

// XXXXX-NEXT:     spirv.func @baz_no_return(%arg0: i32)
// XXXXX-NEXT:       spirv.Return
// XXXXX-NEXT:     }

// XXXXX-NEXT:     spirv.func @baz_no_return_different_control
// XXXXX-NEXT:       spirv.Return
// XXXXX-NEXT:     }

// XXXXX-NEXT:     spirv.func @baz_no_return_another_control
// XXXXX-NEXT:       spirv.Return
// XXXXX-NEXT:     }

// XXXXX-NEXT:     spirv.func @kernel
// XXXXX-NEXT:       spirv.Return
// XXXXX-NEXT:     }

// XXXXX-NEXT:     spirv.func @kernel_different_attr
// XXXXX-NEXT:       spirv.Return
// XXXXX-NEXT:     }
// XXXXX-NEXT:   }
// XXXXX-NEXT:   }

module {
spirv.module Logical GLSL450 {
  spirv.SpecConstant @bar spec_id(5) = 1. : f32

  spirv.func @foo(%arg0: f32) -> (f32) "None" {
    spirv.ReturnValue %arg0 : f32
  }

  spirv.func @foo_duplicate(%arg0: f32) -> (f32) "None" {
    spirv.ReturnValue %arg0 : f32
  }

  spirv.func @foo_different_body(%arg0: f32) -> (f32) "None" {
    %0 = spirv.mlir.referenceof @bar : f32
    spirv.ReturnValue %arg0 : f32
  }

  spirv.func @baz(%arg0: i32) -> (i32) "None" {
    spirv.ReturnValue %arg0 : i32
  }

  spirv.func @baz_no_return(%arg0: i32) "None" {
    spirv.Return
  }

  spirv.func @baz_no_return_duplicate(%arg0: i32) -> () "None" {
    spirv.Return
  }

  spirv.func @baz_no_return_different_control(%arg0: i32) -> () "Inline" {
    spirv.Return
  }

  spirv.func @baz_no_return_another_control(%arg0: i32) -> () "Inline|Pure" {
    spirv.Return
  }

  spirv.func @kernel(
    %arg0: f32,
    %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, CrossWorkgroup>) "None"
  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 1, 1]>} {
    spirv.Return
  }

  spirv.func @kernel_different_attr(
    %arg0: f32,
    %arg1: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32>)>, CrossWorkgroup>) "None"
  attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>} {
    spirv.Return
  }
}
}
