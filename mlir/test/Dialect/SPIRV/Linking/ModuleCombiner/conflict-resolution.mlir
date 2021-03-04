// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

// Test basic renaming of conflicting funcOps.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : f32) -> f32 "None" {
    spv.ReturnValue %arg0 : f32
  }
}
}

// -----

// Test basic renaming of conflicting funcOps across 3 modules.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo_1
// CHECK-NEXT:       spv.FAdd
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo_2
// CHECK-NEXT:       spv.ISub
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : f32) -> f32 "None" {
    %0 = spv.FAdd %arg0, %arg0 : f32
    spv.ReturnValue %0 : f32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    %0 = spv.ISub %arg0, %arg0 : i32
    spv.ReturnValue %0 : i32
  }
}
}

// -----

// Test properly updating references to a renamed funcOp.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @bar
// CHECK-NEXT:       spv.FunctionCall @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : f32) -> f32 "None" {
    spv.ReturnValue %arg0 : f32
  }

  spv.func @bar(%arg0 : f32) -> f32 "None" {
    %0 = spv.FunctionCall @foo(%arg0) : (f32) ->  (f32)
    spv.ReturnValue %0 : f32
  }
}
}

// -----

// Test properly updating references to a renamed funcOp if the functionCallOp
// preceeds the callee funcOp definition.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @bar
// CHECK-NEXT:       spv.FunctionCall @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.func @bar(%arg0 : f32) -> f32 "None" {
    %0 = spv.FunctionCall @foo(%arg0) : (f32) ->  (f32)
    spv.ReturnValue %0 : f32
  }

  spv.func @foo(%arg0 : f32) -> f32 "None" {
    spv.ReturnValue %arg0 : f32
  }
}
}

// -----

// Test properly updating entryPointOp and executionModeOp attached to renamed
// funcOp.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.EntryPoint "GLCompute" @foo_1
// CHECK-NEXT:     spv.ExecutionMode @foo_1 "ContractionOff"
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : f32) -> f32 "None" {
    spv.ReturnValue %arg0 : f32
  }

  spv.EntryPoint "GLCompute" @foo
  spv.ExecutionMode @foo "ContractionOff"
}
}

// -----

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.EntryPoint "GLCompute" @fo
// CHECK-NEXT:     spv.ExecutionMode @foo "ContractionOff"

// CHECK-NEXT:     spv.func @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.EntryPoint "GLCompute" @foo_1
// CHECK-NEXT:     spv.ExecutionMode @foo_1 "ContractionOff"
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
  
  spv.EntryPoint "GLCompute" @foo
  spv.ExecutionMode @foo "ContractionOff"
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : f32) -> f32 "None" {
    spv.ReturnValue %arg0 : f32
  }

  spv.EntryPoint "GLCompute" @foo
  spv.ExecutionMode @foo "ContractionOff"
}
}

// -----

// Resolve conflicting funcOp and globalVariableOp.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.globalVariable @foo_1
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>
}
}

// -----

// Resolve conflicting funcOp and globalVariableOp and update the global variable's
// references.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.globalVariable @foo_1
// CHECK-NEXT:     spv.func @bar
// CHECK-NEXT:       spv.mlir.addressof @foo_1
// CHECK-NEXT:       spv.Load
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>

  spv.func @bar() -> f32 "None" {
    %0 = spv.mlir.addressof @foo : !spv.ptr<f32, Input>
    %1 = spv.Load "Input" %0 : f32
    spv.ReturnValue %1 : f32
  }
}
}

// -----

// Resolve conflicting globalVariableOp and funcOp and update the global variable's
// references.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.globalVariable @foo_1
// CHECK-NEXT:     spv.func @bar
// CHECK-NEXT:       spv.mlir.addressof @foo_1
// CHECK-NEXT:       spv.Load
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>

  spv.func @bar() -> f32 "None" {
    %0 = spv.mlir.addressof @foo : !spv.ptr<f32, Input>
    %1 = spv.Load "Input" %0 : f32
    spv.ReturnValue %1 : f32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}
}

// -----

// Resolve conflicting funcOp and specConstantOp.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.SpecConstant @foo_1
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.SpecConstant @foo = -5 : i32
}
}

// -----

// Resolve conflicting funcOp and specConstantOp and update the spec constant's
// references.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.SpecConstant @foo_1
// CHECK-NEXT:     spv.func @bar
// CHECK-NEXT:       spv.mlir.referenceof @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.SpecConstant @foo = -5 : i32

  spv.func @bar() -> i32 "None" {
    %0 = spv.mlir.referenceof @foo : i32 
    spv.ReturnValue %0 : i32
  }
}
}

// -----

// Resolve conflicting specConstantOp and funcOp and update the spec constant's
// references.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.SpecConstant @foo_1
// CHECK-NEXT:     spv.func @bar
// CHECK-NEXT:       spv.mlir.referenceof @foo_1
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.SpecConstant @foo = -5 : i32

  spv.func @bar() -> i32 "None" {
    %0 = spv.mlir.referenceof @foo : i32
    spv.ReturnValue %0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}
}

// -----

// Resolve conflicting funcOp and specConstantCompositeOp.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.SpecConstant @bar
// CHECK-NEXT:     spv.SpecConstantComposite @foo_1 (@bar, @bar)
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.SpecConstant @bar = -5 : i32
  spv.SpecConstantComposite @foo (@bar, @bar) : !spv.array<2 x i32>
}
}

// -----

// Resolve conflicting funcOp and specConstantCompositeOp and update the spec
// constant's references.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.SpecConstant @bar
// CHECK-NEXT:     spv.SpecConstantComposite @foo_1 (@bar, @bar)
// CHECK-NEXT:     spv.func @baz
// CHECK-NEXT:       spv.mlir.referenceof @foo_1
// CHECK-NEXT:       spv.CompositeExtract
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}

spv.module Logical GLSL450 {
  spv.SpecConstant @bar = -5 : i32
  spv.SpecConstantComposite @foo (@bar, @bar) : !spv.array<2 x i32>

  spv.func @baz() -> i32 "None" {
    %0 = spv.mlir.referenceof @foo : !spv.array<2 x i32>
    %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<2 x i32>
    spv.ReturnValue %1 : i32
  }
}
}

// -----

// Resolve conflicting specConstantCompositeOp and funcOp and update the spec
// constant's references.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.SpecConstant @bar
// CHECK-NEXT:     spv.SpecConstantComposite @foo_1 (@bar, @bar)
// CHECK-NEXT:     spv.func @baz
// CHECK-NEXT:       spv.mlir.referenceof @foo_1
// CHECK-NEXT:       spv.CompositeExtract
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.SpecConstant @bar = -5 : i32
  spv.SpecConstantComposite @foo (@bar, @bar) : !spv.array<2 x i32>

  spv.func @baz() -> i32 "None" {
    %0 = spv.mlir.referenceof @foo : !spv.array<2 x i32>
    %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<2 x i32>
    spv.ReturnValue %1 : i32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }
}
}

// -----

// Resolve conflicting spec constants and funcOps and update the spec constant's
// references.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.SpecConstant @bar_1
// CHECK-NEXT:     spv.SpecConstantComposite @foo_2 (@bar_1, @bar_1)
// CHECK-NEXT:     spv.func @baz
// CHECK-NEXT:       spv.mlir.referenceof @foo_2
// CHECK-NEXT:       spv.CompositeExtract
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @bar
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.SpecConstant @bar = -5 : i32
  spv.SpecConstantComposite @foo (@bar, @bar) : !spv.array<2 x i32>

  spv.func @baz() -> i32 "None" {
    %0 = spv.mlir.referenceof @foo : !spv.array<2 x i32>
    %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<2 x i32>
    spv.ReturnValue %1 : i32
  }
}

spv.module Logical GLSL450 {
  spv.func @foo(%arg0 : i32) -> i32 "None" {
    spv.ReturnValue %arg0 : i32
  }

  spv.func @bar(%arg0 : f32) -> f32 "None" {
    spv.ReturnValue %arg0 : f32
  }
}
}

// -----

// Resolve conflicting globalVariableOps.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.globalVariable @foo_1 bind(1, 0)

// CHECK-NEXT:     spv.globalVariable @foo bind(2, 0)
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>
}

spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(2, 0) : !spv.ptr<f32, Input>
}
}

// -----

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.globalVariable @foo_1 built_in("GlobalInvocationId")

// CHECK-NEXT:     spv.globalVariable @foo built_in("LocalInvocationId")
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.globalVariable @foo built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
}

spv.module Logical GLSL450 {
  spv.globalVariable @foo built_in("LocalInvocationId") : !spv.ptr<vector<3xi32>, Input>
}
}

// -----

// Resolve conflicting globalVariableOp and specConstantOp.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.globalVariable @foo_1

// CHECK-NEXT:     spv.SpecConstant @foo
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>
}

spv.module Logical GLSL450 {
  spv.SpecConstant @foo = -5 : i32
}
}

// -----

// Resolve conflicting specConstantOp and globalVariableOp.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.SpecConstant @foo_1

// CHECK-NEXT:     spv.globalVariable @foo
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.SpecConstant @foo = -5 : i32
}

spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>
}
}

// -----

// Resolve conflicting globalVariableOp and specConstantCompositeOp.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.globalVariable @foo_1

// CHECK-NEXT:     spv.SpecConstant @bar
// CHECK-NEXT:     spv.SpecConstantComposite @foo (@bar, @bar)
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>
}

spv.module Logical GLSL450 {
  spv.SpecConstant @bar = -5 : i32
  spv.SpecConstantComposite @foo (@bar, @bar) : !spv.array<2 x i32>
}
}

// -----

// Resolve conflicting globalVariableOp and specConstantComposite.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.SpecConstant @bar
// CHECK-NEXT:     spv.SpecConstantComposite @foo_1 (@bar, @bar)

// CHECK-NEXT:     spv.globalVariable @foo
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.SpecConstant @bar = -5 : i32
  spv.SpecConstantComposite @foo (@bar, @bar) : !spv.array<2 x i32>
}

spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>
}
}
