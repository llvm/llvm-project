// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

// Multi-way switch routing results through a function variable.

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
// CHECK-LABEL: @switch
  spirv.func @switch(%cond: i32) -> () "None" {
    %zero = spirv.Constant 0 : i32
    %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

// CHECK:        spirv.mlir.selection
    spirv.mlir.selection {
// CHECK-NEXT:     spirv.Switch %{{.*}} : i32, [
// CHECK-NEXT:       default: ^[[DEFAULT:.+]],
// CHECK-NEXT:       2: ^[[CASE2:.+]],
// CHECK-NEXT:       5: ^[[CASE5:.+]]
// CHECK-NEXT:     ]
      spirv.Switch %cond : i32, [
        default: ^default,
        2: ^case2,
        5: ^case5
      ]

// The deserializer emits the target blocks in branch order: default first,
// then the case blocks.
// CHECK-NEXT:   ^[[DEFAULT]]:
// CHECK-NEXT:     spirv.Constant 30
// CHECK-NEXT:     spirv.Store
// CHECK-NEXT:     spirv.Branch ^[[MERGE:.+]]
// CHECK-NEXT:   ^[[CASE2]]:
// CHECK-NEXT:     spirv.Constant 10
// CHECK-NEXT:     spirv.Store
// CHECK-NEXT:     spirv.Branch ^[[MERGE]]
// CHECK-NEXT:   ^[[CASE5]]:
// CHECK-NEXT:     spirv.Constant 20
// CHECK-NEXT:     spirv.Store
// CHECK-NEXT:     spirv.Branch ^[[MERGE]]
// CHECK-NEXT:   ^[[MERGE]]:
// CHECK-NEXT:     spirv.mlir.merge
    ^case2:
      %ten = spirv.Constant 10 : i32
      spirv.Store "Function" %var, %ten : i32
      spirv.Branch ^merge

    ^case5:
      %twenty = spirv.Constant 20 : i32
      spirv.Store "Function" %var, %twenty : i32
      spirv.Branch ^merge

    ^default:
      %thirty = spirv.Constant 30 : i32
      spirv.Store "Function" %var, %thirty : i32
      spirv.Branch ^merge

    ^merge:
      spirv.mlir.merge
    }
    spirv.Return
  }
}

// -----

// Switch with only a default target (no case literals).

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
// CHECK-LABEL: @switch_default_only
  spirv.func @switch_default_only(%cond: i32) -> () "None" {
// CHECK:        spirv.mlir.selection
    spirv.mlir.selection {
// CHECK-NEXT:     spirv.Switch %{{.*}} : i32, [
// CHECK-NEXT:       default: ^[[DEFAULT:.+]]]
      spirv.Switch %cond : i32, [
        default: ^default
      ]

// CHECK-NEXT:   ^[[DEFAULT]]:
    ^default:
// CHECK-NEXT:     spirv.Branch ^[[MERGE:.+]]
      spirv.Branch ^merge

// CHECK-NEXT:   ^[[MERGE]]:
    ^merge:
// CHECK-NEXT:     spirv.mlir.merge
      spirv.mlir.merge
    }
    spirv.Return
  }
}
