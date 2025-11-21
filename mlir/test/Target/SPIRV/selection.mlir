// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

// Selection with both then and else branches

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @selection
  spirv.func @selection(%cond: i1) -> () "None" {
// CHECK-NEXT:   spirv.Constant 0
// CHECK-NEXT:   spirv.Variable
// CHECK:        spirv.Branch ^[[BB:.+]]
// CHECK-NEXT: ^[[BB]]:
    %zero = spirv.Constant 0: i32
    %one = spirv.Constant 1: i32
    %two = spirv.Constant 2: i32
    %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

// CHECK-NEXT:   spirv.mlir.selection control(Flatten)
    spirv.mlir.selection control(Flatten) {
// CHECK-NEXT: spirv.BranchConditional %{{.*}} [5, 10], ^[[THEN:.+]], ^[[ELSE:.+]]
      spirv.BranchConditional %cond [5, 10], ^then, ^else

// CHECK-NEXT:   ^[[THEN]]:
    ^then:
// CHECK-NEXT:     spirv.Constant 1
// CHECK-NEXT:     spirv.Store
      spirv.Store "Function" %var, %one : i32
// CHECK-NEXT:     spirv.Branch ^[[MERGE:.+]]
      spirv.Branch ^merge

// CHECK-NEXT:   ^[[ELSE]]:
    ^else:
// CHECK-NEXT:     spirv.Constant 2
// CHECK-NEXT:     spirv.Store
      spirv.Store "Function" %var, %two : i32
// CHECK-NEXT:     spirv.Branch ^[[MERGE]]
      spirv.Branch ^merge

// CHECK-NEXT:   ^[[MERGE]]:
    ^merge:
// CHECK-NEXT:     spirv.mlir.merge
      spirv.mlir.merge
    }

    spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
  spirv.ExecutionMode @main "LocalSize", 1, 1, 1
}

// -----

// Selection with only then branch
// Selection in function entry block

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: spirv.func @selection
//  CHECK-SAME: (%[[ARG:.*]]: i1)
  spirv.func @selection(%cond: i1) -> (i32) "None" {
// CHECK:        spirv.Branch ^[[BB:.+]]
// CHECK-NEXT: ^[[BB]]:
// CHECK-NEXT:   spirv.mlir.selection
    spirv.mlir.selection {
// CHECK-NEXT: spirv.BranchConditional %[[ARG]], ^[[THEN:.+]], ^[[ELSE:.+]]
      spirv.BranchConditional %cond, ^then, ^merge

// CHECK:        ^[[THEN]]:
    ^then:
      %zero = spirv.Constant 0 : i32
      spirv.ReturnValue  %zero : i32

// CHECK:        ^[[ELSE]]:
    ^merge:
// CHECK-NEXT:     spirv.mlir.merge
      spirv.mlir.merge
    }

    %one = spirv.Constant 1 : i32
    spirv.ReturnValue  %one : i32
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
  spirv.ExecutionMode @main "LocalSize", 1, 1, 1
}

// -----

// Selection with control flow afterwards
// SSA value def before selection and use after selection

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @selection_cf()
  spirv.func @selection_cf() -> () "None" {
    %true = spirv.Constant true
    %false = spirv.Constant false
    %zero = spirv.Constant 0 : i32
    %one = spirv.Constant 1 : i32
// CHECK-NEXT:    %[[VAR:.+]] = spirv.Variable
    %var = spirv.Variable : !spirv.ptr<i1, Function>
// CHECK-NEXT:    spirv.Branch ^[[BB:.+]]
// CHECK-NEXT:  ^[[BB]]:
// CHECK:    spirv.Branch ^[[BB:.+]]
// CHECK-NEXT:  ^[[BB]]:

// CHECK-NEXT:    spirv.mlir.selection {
    spirv.mlir.selection {
//      CHECK:      spirv.BranchConditional %{{.+}}, ^[[THEN0:.+]], ^[[ELSE0:.+]]
      spirv.BranchConditional %true, ^then0, ^else0

// CHECK-NEXT:    ^[[THEN0]]:
//      CHECK:      spirv.Store "Function" %[[VAR]]
// CHECK-NEXT:      spirv.Branch ^[[MERGE:.+]]
    ^then0:
      spirv.Store "Function" %var, %true : i1
      spirv.Branch ^merge

// CHECK-NEXT:    ^[[ELSE0]]:
//      CHECK:      spirv.Store "Function" %[[VAR]]
// CHECK-NEXT:      spirv.Branch ^[[MERGE]]
    ^else0:
      spirv.Store "Function" %var, %false : i1
      spirv.Branch ^merge

// CHECK-NEXT:    ^[[MERGE]]:
// CHECK-NEXT:      spirv.mlir.merge
    ^merge:
      spirv.mlir.merge
// CHECK-NEXT:    }
    }

// CHECK-NEXT:    spirv.Load "Function" %[[VAR]]
    %cond = spirv.Load "Function" %var : i1
    spirv.mlir.selection {
//      CHECK:    spirv.BranchConditional %1, ^[[THEN1:.+]](%{{.+}} : i32), ^[[ELSE1:.+]](%{{.+}}, %{{.+}} : i32, i32)
      spirv.BranchConditional %cond, ^then1(%one: i32), ^else1(%zero, %zero: i32, i32)

// CHECK-NEXT:  ^[[THEN1]](%{{.+}}: i32):
// CHECK-NEXT:    spirv.Return
    ^then1(%arg0: i32):
      spirv.Return

// CHECK-NEXT:  ^[[ELSE1]](%{{.+}}: i32, %{{.+}}: i32):
// CHECK-NEXT:    spirv.Return
    ^else1(%arg1: i32, %arg2: i32):
      spirv.Return
    ^merge:
      spirv.mlir.merge
    }

    spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
  spirv.ExecutionMode @main "LocalSize", 1, 1, 1
}

// -----

// Selection yielding values

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @selection_yield
  spirv.func @selection_yield(%cond: i1) -> () "None" {
// CHECK-NEXT:   spirv.Constant 0
// CHECK-NEXT:   spirv.Variable
    %zero = spirv.Constant 0 : i32
    %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

// CHECK:        spirv.Branch ^[[BB:.+]]
// CHECK-NEXT: ^[[BB]]:
// CHECK-NEXT:   {{%.*}} = spirv.mlir.selection -> i32
    %yield = spirv.mlir.selection -> i32 {
// CHECK-NEXT: spirv.BranchConditional %{{.*}} [5, 10], ^[[THEN:.+]], ^[[ELSE:.+]]
      spirv.BranchConditional %cond [5, 10], ^then, ^else

// CHECK-NEXT:   ^[[THEN]]:
    ^then:
// CHECK-NEXT:     spirv.Constant 1
      %one = spirv.Constant 1: i32

// CHECK-NEXT:     spirv.Branch ^[[MERGE:.+]]({{%.*}} : i32)
      spirv.Branch ^merge(%one : i32)

// CHECK-NEXT:   ^[[ELSE]]:
    ^else:
// CHECK-NEXT:     spirv.Constant 2
      %two = spirv.Constant 2: i32
// CHECK-NEXT:     spirv.Branch ^[[MERGE]]({{%.*}} : i32)
      spirv.Branch ^merge(%two : i32)

// CHECK-NEXT:   ^[[MERGE]]({{%.*}}: i32):
    ^merge(%merged: i32):
// CHECK-NEXT:     spirv.mlir.merge {{%.*}} : i32
      spirv.mlir.merge %merged : i32
    }

// CHECK:        spirv.Store "Function" {{%.*}}, {{%.*}} : i32
    spirv.Store "Function" %var, %yield : i32

    spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
  spirv.ExecutionMode @main "LocalSize", 1, 1, 1
}

// -----

// Selection with switch

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @selection_switch
  spirv.func @selection_switch(%selector: i32) -> () "None" {
    %zero = spirv.Constant 0: i32
    %one = spirv.Constant 1: i32
    %two = spirv.Constant 2: i32
    %three = spirv.Constant 3: i32
    %four = spirv.Constant 4: i32
// CHECK: {{%.*}} = spirv.Variable init({{%.*}}) : !spirv.ptr<i32, Function>
    %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
// CHECK: spirv.mlir.selection {
    spirv.mlir.selection {
// CHECK-NEXT: spirv.Switch {{%.*}} : i32, [
// CHECK-NEXT: default: ^[[DEFAULT:.+]],
// CHECK-NEXT: 0: ^[[CASE0:.+]],
// CHECK-NEXT: 1: ^[[CASE1:.+]],
// CHECK-NEXT: 2: ^[[CASE2:.+]]
      spirv.Switch %selector : i32, [
        default: ^default,
        0: ^case0,
        1: ^case1,
        2: ^case2
      ]
// CHECK: ^[[DEFAULT]]
    ^default:
// CHECK: spirv.Store "Function" {{%.*}}, {{%.*}} : i32
      spirv.Store "Function" %var, %one : i32
// CHECK-NEXT: spirv.Branch ^[[MERGE:.+]]
      spirv.Branch ^merge
// CHECK-NEXT: ^[[CASE0]]
    ^case0:
// CHECK: spirv.Store "Function" {{%.*}}, {{%.*}} : i32
      spirv.Store "Function" %var, %two : i32
// CHECK-NEXT: spirv.Branch ^[[MERGE:.+]]
      spirv.Branch ^merge
// CHECK-NEXT: ^[[CASE1]]
    ^case1:
// CHECK: spirv.Store "Function" {{%.*}}, {{%.*}} : i32
      spirv.Store "Function" %var, %three : i32
// CHECK-NEXT: spirv.Branch ^[[MERGE:.+]]
      spirv.Branch ^merge
// CHECK-NEXT: ^[[CASE2]]
    ^case2:
// CHECK: spirv.Store "Function" {{%.*}}, {{%.*}} : i32
      spirv.Store "Function" %var, %four : i32
// CHECK-NEXT: spirv.Branch ^[[MERGE:.+]]
      spirv.Branch ^merge
// CHECK-NEXT: ^[[MERGE]]
    ^merge:
// CHECK-NEXT: spirv.mlir.merge
      spirv.mlir.merge
// CHECK-NEXT: }
    }
// CHECK-NEXT: spirv.Return
    spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
  spirv.ExecutionMode @main "LocalSize", 1, 1, 1
}
