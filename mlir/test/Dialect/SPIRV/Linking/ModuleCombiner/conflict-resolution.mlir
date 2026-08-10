// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

// Test basic renaming of conflicting funcOps.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : f32) -> f32 "None" {
    spirv.ReturnValue %arg0 : f32
  }
}
}

// -----

// Test basic renaming of conflicting funcOps across 3 modules.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo_1
// CHECK-NEXT:       spirv.FAdd
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo_2
// CHECK-NEXT:       spirv.ISub
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : f32) -> f32 "None" {
    %0 = spirv.FAdd %arg0, %arg0 : f32
    spirv.ReturnValue %0 : f32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    %0 = spirv.ISub %arg0, %arg0 : i32
    spirv.ReturnValue %0 : i32
  }
}
}

// -----

// Test properly updating references to a renamed funcOp.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @bar
// CHECK-NEXT:       spirv.FunctionCall @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : f32) -> f32 "None" {
    spirv.ReturnValue %arg0 : f32
  }

  spirv.func @bar(%arg0 : f32) -> f32 "None" {
    %0 = spirv.FunctionCall @foo(%arg0) : (f32) ->  (f32)
    spirv.ReturnValue %0 : f32
  }
}
}

// -----

// Test properly updating references to a renamed funcOp if the functionCallOp
// preceeds the callee funcOp definition.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @bar
// CHECK-NEXT:       spirv.FunctionCall @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @bar(%arg0 : f32) -> f32 "None" {
    %0 = spirv.FunctionCall @foo(%arg0) : (f32) ->  (f32)
    spirv.ReturnValue %0 : f32
  }

  spirv.func @foo(%arg0 : f32) -> f32 "None" {
    spirv.ReturnValue %arg0 : f32
  }
}
}

// -----

// Test properly updating entryPointOp and executionModeOp attached to renamed
// funcOp.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.EntryPoint "GLCompute" @foo_1
// CHECK-NEXT:     spirv.ExecutionMode @foo_1 "ContractionOff"
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : f32) -> f32 "None" {
    spirv.ReturnValue %arg0 : f32
  }

  spirv.EntryPoint "GLCompute" @foo
  spirv.ExecutionMode @foo "ContractionOff"
}
}

// -----

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.EntryPoint "GLCompute" @fo
// CHECK-NEXT:     spirv.ExecutionMode @foo "ContractionOff"

// CHECK-NEXT:     spirv.func @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.EntryPoint "GLCompute" @foo_1
// CHECK-NEXT:     spirv.ExecutionMode @foo_1 "ContractionOff"
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }

  spirv.EntryPoint "GLCompute" @foo
  spirv.ExecutionMode @foo "ContractionOff"
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : f32) -> f32 "None" {
    spirv.ReturnValue %arg0 : f32
  }

  spirv.EntryPoint "GLCompute" @foo
  spirv.ExecutionMode @foo "ContractionOff"
}
}

// -----

// Resolve conflicting funcOp and globalVariableOp.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.GlobalVariable @foo_1
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>
}
}

// -----

// Resolve conflicting funcOp and globalVariableOp and update the global variable's
// references.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.GlobalVariable @foo_1
// CHECK-NEXT:     spirv.func @bar
// CHECK-NEXT:       spirv.mlir.addressof @foo_1
// CHECK-NEXT:       spirv.Load
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>

  spirv.func @bar() -> f32 "None" {
    %0 = spirv.mlir.addressof @foo : !spirv.ptr<f32, Input>
    %1 = spirv.Load "Input" %0 : f32
    spirv.ReturnValue %1 : f32
  }
}
}

// -----

// Resolve conflicting globalVariableOp and funcOp and update the global variable's
// references.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.GlobalVariable @foo_1
// CHECK-NEXT:     spirv.func @bar
// CHECK-NEXT:       spirv.mlir.addressof @foo_1
// CHECK-NEXT:       spirv.Load
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>

  spirv.func @bar() -> f32 "None" {
    %0 = spirv.mlir.addressof @foo : !spirv.ptr<f32, Input>
    %1 = spirv.Load "Input" %0 : f32
    spirv.ReturnValue %1 : f32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}
}

// -----

// Resolve conflicting funcOp and specConstantOp.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.SpecConstant @foo_1
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.SpecConstant @foo = -5 : i32
}
}

// -----

// Resolve conflicting funcOp and specConstantOp and update the spec constant's
// references.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.SpecConstant @foo_1
// CHECK-NEXT:     spirv.func @bar
// CHECK-NEXT:       spirv.mlir.referenceof @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.SpecConstant @foo = -5 : i32

  spirv.func @bar() -> i32 "None" {
    %0 = spirv.mlir.referenceof @foo : i32
    spirv.ReturnValue %0 : i32
  }
}
}

// -----

// Resolve conflicting specConstantOp and funcOp and update the spec constant's
// references.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.SpecConstant @foo_1
// CHECK-NEXT:     spirv.func @bar
// CHECK-NEXT:       spirv.mlir.referenceof @foo_1
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.SpecConstant @foo = -5 : i32

  spirv.func @bar() -> i32 "None" {
    %0 = spirv.mlir.referenceof @foo : i32
    spirv.ReturnValue %0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}
}

// -----

// Resolve conflicting funcOp and specConstantCompositeOp.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.SpecConstant @bar
// CHECK-NEXT:     spirv.SpecConstantComposite @foo_1 (@bar, @bar)
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.SpecConstant @bar = -5 : i32
  spirv.SpecConstantComposite @foo (@bar, @bar) : !spirv.array<2 x i32>
}
}

// -----

// Resolve conflicting funcOp and specConstantCompositeOp and update the spec
// constant's references.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.SpecConstant @bar
// CHECK-NEXT:     spirv.SpecConstantComposite @foo_1 (@bar, @bar)
// CHECK-NEXT:     spirv.func @baz
// CHECK-NEXT:       spirv.mlir.referenceof @foo_1
// CHECK-NEXT:       spirv.CompositeExtract
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.SpecConstant @bar = -5 : i32
  spirv.SpecConstantComposite @foo (@bar, @bar) : !spirv.array<2 x i32>

  spirv.func @baz() -> i32 "None" {
    %0 = spirv.mlir.referenceof @foo : !spirv.array<2 x i32>
    %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.array<2 x i32>
    spirv.ReturnValue %1 : i32
  }
}
}

// -----

// Resolve conflicting specConstantCompositeOp and funcOp and update the spec
// constant's references.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.SpecConstant @bar
// CHECK-NEXT:     spirv.SpecConstantComposite @foo_1 (@bar, @bar)
// CHECK-NEXT:     spirv.func @baz
// CHECK-NEXT:       spirv.mlir.referenceof @foo_1
// CHECK-NEXT:       spirv.CompositeExtract
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.SpecConstant @bar = -5 : i32
  spirv.SpecConstantComposite @foo (@bar, @bar) : !spirv.array<2 x i32>

  spirv.func @baz() -> i32 "None" {
    %0 = spirv.mlir.referenceof @foo : !spirv.array<2 x i32>
    %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.array<2 x i32>
    spirv.ReturnValue %1 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }
}
}

// -----

// Resolve conflicting spec constants and funcOps and update the spec constant's
// references.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.SpecConstant @bar_1
// CHECK-NEXT:     spirv.SpecConstantComposite @foo_2 (@bar_1, @bar_1)
// CHECK-NEXT:     spirv.func @baz
// CHECK-NEXT:       spirv.mlir.referenceof @foo_2
// CHECK-NEXT:       spirv.CompositeExtract
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @foo
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spirv.func @bar
// CHECK-NEXT:       spirv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.SpecConstant @bar = -5 : i32
  spirv.SpecConstantComposite @foo (@bar, @bar) : !spirv.array<2 x i32>

  spirv.func @baz() -> i32 "None" {
    %0 = spirv.mlir.referenceof @foo : !spirv.array<2 x i32>
    %1 = spirv.CompositeExtract %0[0 : i32] : !spirv.array<2 x i32>
    spirv.ReturnValue %1 : i32
  }
}

spirv.module Logical GLSL450 {
  spirv.func @foo(%arg0 : i32) -> i32 "None" {
    spirv.ReturnValue %arg0 : i32
  }

  spirv.func @bar(%arg0 : f32) -> f32 "None" {
    spirv.ReturnValue %arg0 : f32
  }
}
}

// -----

// Resolve conflicting globalVariableOps.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.GlobalVariable @foo_1 bind(1, 0)

// CHECK-NEXT:     spirv.GlobalVariable @foo bind(2, 0)
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(2, 0) : !spirv.ptr<f32, Input>
}
}

// -----

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.GlobalVariable @foo_1 built_in("GlobalInvocationId")

// CHECK-NEXT:     spirv.GlobalVariable @foo built_in("LocalInvocationId")
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
}
}

// -----

// Resolve conflicting globalVariableOp and specConstantOp.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.GlobalVariable @foo_1

// CHECK-NEXT:     spirv.SpecConstant @foo
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>
}

spirv.module Logical GLSL450 {
  spirv.SpecConstant @foo = -5 : i32
}
}

// -----

// Resolve conflicting specConstantOp and globalVariableOp.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.SpecConstant @foo_1

// CHECK-NEXT:     spirv.GlobalVariable @foo
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.SpecConstant @foo = -5 : i32
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>
}
}

// -----

// Resolve conflicting globalVariableOp and specConstantCompositeOp.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.GlobalVariable @foo_1

// CHECK-NEXT:     spirv.SpecConstant @bar
// CHECK-NEXT:     spirv.SpecConstantComposite @foo (@bar, @bar)
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>
}

spirv.module Logical GLSL450 {
  spirv.SpecConstant @bar = -5 : i32
  spirv.SpecConstantComposite @foo (@bar, @bar) : !spirv.array<2 x i32>
}
}

// -----

// Resolve conflicting globalVariableOp and specConstantComposite.

// CHECK:      module {
// CHECK-NEXT:   spirv.module Logical GLSL450 {
// CHECK-NEXT:     spirv.SpecConstant @bar
// CHECK-NEXT:     spirv.SpecConstantComposite @foo_1 (@bar, @bar)

// CHECK-NEXT:     spirv.GlobalVariable @foo
// CHECK-NEXT: }

module {
spirv.module Logical GLSL450 {
  spirv.SpecConstant @bar = -5 : i32
  spirv.SpecConstantComposite @foo (@bar, @bar) : !spirv.array<2 x i32>
}

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @foo bind(1, 0) : !spirv.ptr<f32, Input>
}
}
