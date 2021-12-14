// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// Selection with both then and else branches

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @selection
  spv.func @selection(%cond: i1) -> () "None" {
// CHECK-NEXT:   spv.Constant 0
// CHECK-NEXT:   spv.Variable
// CHECK:        spv.Branch ^[[BB:.+]]
// CHECK-NEXT: ^[[BB]]:
    %zero = spv.Constant 0: i32
    %one = spv.Constant 1: i32
    %two = spv.Constant 2: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

// CHECK-NEXT:   spv.mlir.selection control(Flatten)
    spv.mlir.selection control(Flatten) {
// CHECK-NEXT: spv.BranchConditional %{{.*}} [5, 10], ^[[THEN:.+]], ^[[ELSE:.+]]
      spv.BranchConditional %cond [5, 10], ^then, ^else

// CHECK-NEXT:   ^[[THEN]]:
    ^then:
// CHECK-NEXT:     spv.Constant 1
// CHECK-NEXT:     spv.Store
      spv.Store "Function" %var, %one : i32
// CHECK-NEXT:     spv.Branch ^[[MERGE:.+]]
      spv.Branch ^merge

// CHECK-NEXT:   ^[[ELSE]]:
    ^else:
// CHECK-NEXT:     spv.Constant 2
// CHECK-NEXT:     spv.Store
      spv.Store "Function" %var, %two : i32
// CHECK-NEXT:     spv.Branch ^[[MERGE]]
      spv.Branch ^merge

// CHECK-NEXT:   ^[[MERGE]]:
    ^merge:
// CHECK-NEXT:     spv.mlir.merge
      spv.mlir.merge
    }

    spv.Return
  }

  spv.func @main() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
  spv.ExecutionMode @main "LocalSize", 1, 1, 1
}

// -----

// Selection with only then branch
// Selection in function entry block

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: spv.func @selection
//  CHECK-SAME: (%[[ARG:.*]]: i1)
  spv.func @selection(%cond: i1) -> (i32) "None" {
// CHECK:        spv.Branch ^[[BB:.+]]
// CHECK-NEXT: ^[[BB]]:
// CHECK-NEXT:   spv.mlir.selection
    spv.mlir.selection {
// CHECK-NEXT: spv.BranchConditional %[[ARG]], ^[[THEN:.+]], ^[[ELSE:.+]]
      spv.BranchConditional %cond, ^then, ^merge

// CHECK:        ^[[THEN]]:
    ^then:
      %zero = spv.Constant 0 : i32
      spv.ReturnValue  %zero : i32

// CHECK:        ^[[ELSE]]:
    ^merge:
// CHECK-NEXT:     spv.mlir.merge
      spv.mlir.merge
    }

    %one = spv.Constant 1 : i32
    spv.ReturnValue  %one : i32
  }

  spv.func @main() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
  spv.ExecutionMode @main "LocalSize", 1, 1, 1
}

// -----

// Selection with control flow afterwards
// SSA value def before selection and use after selection

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @selection_cf()
  spv.func @selection_cf() -> () "None" {
    %true = spv.Constant true
    %false = spv.Constant false
    %zero = spv.Constant 0 : i32
    %one = spv.Constant 1 : i32
// CHECK-NEXT:    %[[VAR:.+]] = spv.Variable
    %var = spv.Variable : !spv.ptr<i1, Function>
// CHECK-NEXT:    spv.Branch ^[[BB:.+]]
// CHECK-NEXT:  ^[[BB]]:

// CHECK-NEXT:    spv.mlir.selection {
    spv.mlir.selection {
//      CHECK:      spv.BranchConditional %{{.+}}, ^[[THEN0:.+]], ^[[ELSE0:.+]]
      spv.BranchConditional %true, ^then0, ^else0

// CHECK-NEXT:    ^[[THEN0]]:
//      CHECK:      spv.Store "Function" %[[VAR]]
// CHECK-NEXT:      spv.Branch ^[[MERGE:.+]]
    ^then0:
      spv.Store "Function" %var, %true : i1
      spv.Branch ^merge

// CHECK-NEXT:    ^[[ELSE0]]:
//      CHECK:      spv.Store "Function" %[[VAR]]
// CHECK-NEXT:      spv.Branch ^[[MERGE]]
    ^else0:
      spv.Store "Function" %var, %false : i1
      spv.Branch ^merge

// CHECK-NEXT:    ^[[MERGE]]:
// CHECK-NEXT:      spv.mlir.merge
    ^merge:
      spv.mlir.merge
// CHECK-NEXT:    }
    }

// CHECK-NEXT:    spv.Load "Function" %[[VAR]]
    %cond = spv.Load "Function" %var : i1
//      CHECK:    spv.BranchConditional %1, ^[[THEN1:.+]](%{{.+}} : i32), ^[[ELSE1:.+]](%{{.+}}, %{{.+}} : i32, i32)
    spv.BranchConditional %cond, ^then1(%one: i32), ^else1(%zero, %zero: i32, i32)

// CHECK-NEXT:  ^[[THEN1]](%{{.+}}: i32):
// CHECK-NEXT:    spv.Return
  ^then1(%arg0: i32):
    spv.Return

// CHECK-NEXT:  ^[[ELSE1]](%{{.+}}: i32, %{{.+}}: i32):
// CHECK-NEXT:    spv.Return
  ^else1(%arg1: i32, %arg2: i32):
    spv.Return
  }
}
