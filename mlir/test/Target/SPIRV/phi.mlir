// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

// Test branch with one block argument

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "None" {
// CHECK:        %[[CST:.*]] = spirv.Constant 0
    %zero = spirv.Constant 0 : i32
// CHECK-NEXT:   spirv.Branch ^bb1(%[[CST]] : i32)
    spirv.Branch ^bb1(%zero : i32)
// CHECK-NEXT: ^bb1(%{{.*}}: i32):
  ^bb1(%arg0: i32):
   spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}

// -----

// Test branch with multiple block arguments

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "None" {
// CHECK:        %[[ZERO:.*]] = spirv.Constant 0
    %zero = spirv.Constant 0 : i32
// CHECK-NEXT:   %[[ONE:.*]] = spirv.Constant 1
    %one = spirv.Constant 1.0 : f32
// CHECK-NEXT:   spirv.Branch ^bb1(%[[ZERO]], %[[ONE]] : i32, f32)
    spirv.Branch ^bb1(%zero, %one : i32, f32)

// CHECK-NEXT: ^bb1(%{{.*}}: i32, %{{.*}}: f32):     // pred: ^bb0
  ^bb1(%arg0: i32, %arg1: f32):
   spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}

// -----

// Test using block arguments within branch

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "None" {
// CHECK:        %[[CST0:.*]] = spirv.Constant 0
    %zero = spirv.Constant 0 : i32
// CHECK-NEXT:   spirv.Branch ^bb1(%[[CST0]] : i32)
    spirv.Branch ^bb1(%zero : i32)

// CHECK-NEXT: ^bb1(%[[ARG:.*]]: i32):
  ^bb1(%arg0: i32):
// CHECK-NEXT:   %[[ADD:.*]] = spirv.IAdd %[[ARG]], %[[ARG]] : i32
    %0 = spirv.IAdd %arg0, %arg0 : i32
// CHECK-NEXT:   %[[CST1:.*]] = spirv.Constant 0
// CHECK-NEXT:   spirv.Branch ^bb2(%[[CST1]], %[[ADD]] : i32, i32)
    spirv.Branch ^bb2(%zero, %0 : i32, i32)

// CHECK-NEXT: ^bb2(%{{.*}}: i32, %{{.*}}: i32):
  ^bb2(%arg1: i32, %arg2: i32):
   spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}

// -----

// Test block not following domination order

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "None" {
// CHECK:        spirv.Branch ^bb1
    spirv.Branch ^bb1

// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   %[[ZERO:.*]] = spirv.Constant 0
// CHECK-NEXT:   %[[ONE:.*]] = spirv.Constant 1
// CHECK-NEXT:   spirv.Branch ^bb2(%[[ZERO]], %[[ONE]] : i32, f32)

// CHECK-NEXT: ^bb2(%{{.*}}: i32, %{{.*}}: f32):
  ^bb2(%arg0: i32, %arg1: f32):
// CHECK-NEXT:   spirv.Return
   spirv.Return

  // This block is reordered to follow domination order.
  ^bb1:
    %zero = spirv.Constant 0 : i32
    %one = spirv.Constant 1.0 : f32
    spirv.Branch ^bb2(%zero, %one : i32, f32)
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}

// -----

// Test multiple predecessors

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "None" {
    %var = spirv.Variable : !spirv.ptr<i32, Function>

// CHECK:      spirv.mlir.selection
    spirv.mlir.selection {
      %true = spirv.Constant true
// CHECK:        spirv.BranchConditional %{{.*}}, ^bb1, ^bb2
      spirv.BranchConditional %true, ^true, ^false

// CHECK-NEXT: ^bb1:
    ^true:
// CHECK-NEXT:   %[[ZERO:.*]] = spirv.Constant 0
      %zero = spirv.Constant 0 : i32
// CHECK-NEXT:   spirv.Branch ^bb3(%[[ZERO]] : i32)
      spirv.Branch ^phi(%zero: i32)

// CHECK-NEXT: ^bb2:
    ^false:
// CHECK-NEXT:   %[[ONE:.*]] = spirv.Constant 1
      %one = spirv.Constant 1 : i32
// CHECK-NEXT:   spirv.Branch ^bb3(%[[ONE]] : i32)
      spirv.Branch ^phi(%one: i32)

// CHECK-NEXT: ^bb3(%[[ARG:.*]]: i32):
    ^phi(%arg: i32):
// CHECK-NEXT:   spirv.Store "Function" %{{.*}}, %[[ARG]] : i32
      spirv.Store "Function" %var, %arg : i32
// CHECK-NEXT:   spirv.Return
      spirv.Return

// CHECK-NEXT: ^bb4:
    ^merge:
// CHECK-NEXT:   spirv.mlir.merge
      spirv.mlir.merge
    }
    spirv.Return
  }

  spirv.func @main() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main
}

// -----

// Test nested loops with block arguments

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.GlobalVariable @__builtin_var_NumWorkgroups__ built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.func @fmul_kernel() "None" {
    %3 = spirv.Constant 12 : i32
    %4 = spirv.Constant 32 : i32
    %5 = spirv.Constant 4 : i32
    %6 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
    %7 = spirv.Load "Input" %6 : vector<3xi32>
    %8 = spirv.CompositeExtract %7[0 : i32] : vector<3xi32>
    %9 = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
    %10 = spirv.Load "Input" %9 : vector<3xi32>
    %11 = spirv.CompositeExtract %10[1 : i32] : vector<3xi32>
    %18 = spirv.mlir.addressof @__builtin_var_NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
    %19 = spirv.Load "Input" %18 : vector<3xi32>
    %20 = spirv.CompositeExtract %19[0 : i32] : vector<3xi32>
    %21 = spirv.mlir.addressof @__builtin_var_NumWorkgroups__ : !spirv.ptr<vector<3xi32>, Input>
    %22 = spirv.Load "Input" %21 : vector<3xi32>
    %23 = spirv.CompositeExtract %22[1 : i32] : vector<3xi32>
    %30 = spirv.IMul %11, %4 : i32
    %31 = spirv.IMul %23, %4 : i32

// CHECK:   spirv.Branch ^[[FN_BB:.*]](%{{.*}} : i32)
// CHECK: ^[[FN_BB]](%[[FN_BB_ARG:.*]]: i32):
// CHECK:   spirv.mlir.loop {
    spirv.mlir.loop {
// CHECK:     spirv.Branch ^bb1(%[[FN_BB_ARG]] : i32)
      spirv.Branch ^bb1(%30 : i32)
// CHECK:   ^[[LP1_HDR:.*]](%[[LP1_HDR_ARG:.*]]: i32):
    ^bb1(%32: i32):
// CHECK:     spirv.SLessThan
      %33 = spirv.SLessThan %32, %3 : i32
// CHECK:     spirv.BranchConditional %{{.*}}, ^[[LP1_BDY:.*]], ^[[LP1_MG:.*]]
      spirv.BranchConditional %33, ^bb2, ^bb3
// CHECK:   ^[[LP1_BDY]]:
    ^bb2:
// CHECK:     %[[MUL:.*]] = spirv.IMul
      %34 = spirv.IMul %8, %5 : i32
// CHECK:     spirv.IMul
      %35 = spirv.IMul %20, %5 : i32
// CHECK:     spirv.Branch ^[[LP1_CNT:.*]](%[[MUL]] : i32)
// CHECK:   ^[[LP1_CNT]](%[[LP1_CNT_ARG:.*]]: i32):
// CHECK:     spirv.mlir.loop {
      spirv.mlir.loop {
// CHECK:       spirv.Branch ^[[LP2_HDR:.*]](%[[LP1_CNT_ARG]] : i32)
        spirv.Branch ^bb1(%34 : i32)
// CHECK:     ^[[LP2_HDR]](%[[LP2_HDR_ARG:.*]]: i32):
      ^bb1(%37: i32):
// CHECK:       spirv.SLessThan %[[LP2_HDR_ARG]]
        %38 = spirv.SLessThan %37, %5 : i32
// CHECK:       spirv.BranchConditional %{{.*}}, ^[[LP2_BDY:.*]], ^[[LP2_MG:.*]]
        spirv.BranchConditional %38, ^bb2, ^bb3
// CHECK:     ^[[LP2_BDY]]:
      ^bb2:
// CHECK:       %[[ADD1:.*]] = spirv.IAdd
        %48 = spirv.IAdd %37, %35 : i32
// CHECK:       spirv.Branch ^[[LP2_HDR]](%[[ADD1]] : i32)
        spirv.Branch ^bb1(%48 : i32)
// CHECK:     ^[[LP2_MG]]:
      ^bb3:
// CHECK:       spirv.mlir.merge
        spirv.mlir.merge
      }
// CHECK:     %[[ADD2:.*]] = spirv.IAdd %[[LP1_HDR_ARG]]
      %36 = spirv.IAdd %32, %31 : i32
// CHECK:     spirv.Branch ^[[LP1_HDR]](%[[ADD2]] : i32)
      spirv.Branch ^bb1(%36 : i32)
// CHECK:   ^[[LP1_MG]]:
    ^bb3:
// CHECK:     spirv.mlir.merge
      spirv.mlir.merge
    }
    spirv.Return
  }

  spirv.EntryPoint "GLCompute" @fmul_kernel, @__builtin_var_WorkgroupId__, @__builtin_var_NumWorkgroups__
  spirv.ExecutionMode @fmul_kernel "LocalSize", 32, 1, 1
}

// -----

// Test back-to-back loops with block arguments

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @fmul_kernel() "None" {
    %cst4 = spirv.Constant 4 : i32

    %val1 = spirv.Constant 43 : i32
    %val2 = spirv.Constant 44 : i32

// CHECK:        spirv.Constant 43
// CHECK-NEXT:   spirv.Branch ^[[BB1:.+]](%{{.+}} : i32)
// CHECK-NEXT: ^[[BB1]](%{{.+}}: i32):
// CHECK-NEXT:   spirv.mlir.loop
    spirv.mlir.loop { // loop 1
      spirv.Branch ^bb1(%val1 : i32)
    ^bb1(%loop1_bb_arg: i32):
      %loop1_lt = spirv.SLessThan %loop1_bb_arg, %cst4 : i32
      spirv.BranchConditional %loop1_lt, ^bb2, ^bb3
    ^bb2:
      %loop1_add = spirv.IAdd %loop1_bb_arg, %cst4 : i32
      spirv.Branch ^bb1(%loop1_add : i32)
    ^bb3:
      spirv.mlir.merge
    }

// CHECK:        spirv.Constant 44
// CHECK-NEXT:   spirv.Branch ^[[BB2:.+]](%{{.+}} : i32)
// CHECK-NEXT: ^[[BB2]](%{{.+}}: i32):
// CHECK-NEXT:   spirv.mlir.loop
    spirv.mlir.loop { // loop 2
      spirv.Branch ^bb1(%val2 : i32)
    ^bb1(%loop2_bb_arg: i32):
      %loop2_lt = spirv.SLessThan %loop2_bb_arg, %cst4 : i32
      spirv.BranchConditional %loop2_lt, ^bb2, ^bb3
    ^bb2:
      %loop2_add = spirv.IAdd %loop2_bb_arg, %cst4 : i32
      spirv.Branch ^bb1(%loop2_add : i32)
    ^bb3:
      spirv.mlir.merge
    }

    spirv.Return
  }

  spirv.EntryPoint "GLCompute" @fmul_kernel
  spirv.ExecutionMode @fmul_kernel "LocalSize", 32, 1, 1
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @cond_branch_true_argument
  spirv.func @cond_branch_true_argument() -> () "None" {
    %true = spirv.Constant true
    %zero = spirv.Constant 0 : i32
    %one = spirv.Constant 1 : i32
// CHECK:   spirv.BranchConditional %{{.*}}, ^[[true1:.*]](%{{.*}}, %{{.*}} : i32, i32), ^[[false1:.*]]
    spirv.BranchConditional %true, ^true1(%zero, %zero: i32, i32), ^false1
// CHECK: [[true1]](%{{.*}}: i32, %{{.*}}: i32)
  ^true1(%arg0: i32, %arg1: i32):
    spirv.Return
// CHECK: [[false1]]:
  ^false1:
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @cond_branch_false_argument
  spirv.func @cond_branch_false_argument() -> () "None" {
    %true = spirv.Constant true
    %zero = spirv.Constant 0 : i32
    %one = spirv.Constant 1 : i32
// CHECK:   spirv.BranchConditional %{{.*}}, ^[[true1:.*]], ^[[false1:.*]](%{{.*}}, %{{.*}} : i32, i32)
    spirv.BranchConditional %true, ^true1, ^false1(%zero, %zero: i32, i32)
// CHECK: [[true1]]:
  ^true1:
    spirv.Return
// CHECK: [[false1]](%{{.*}}: i32, %{{.*}}: i32):
  ^false1(%arg0: i32, %arg1: i32):
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
// CHECK-LABEL: @cond_branch_true_and_false_argument
  spirv.func @cond_branch_true_and_false_argument() -> () "None" {
    %true = spirv.Constant true
    %zero = spirv.Constant 0 : i32
    %one = spirv.Constant 1 : i32
// CHECK:   spirv.BranchConditional %{{.*}}, ^[[true1:.*]](%{{.*}} : i32), ^[[false1:.*]](%{{.*}}, %{{.*}} : i32, i32)
    spirv.BranchConditional %true, ^true1(%one: i32), ^false1(%zero, %zero: i32, i32)
// CHECK: [[true1]](%{{.*}}: i32):
  ^true1(%arg0: i32):
    spirv.Return
// CHECK: [[false1]](%{{.*}}: i32, %{{.*}}: i32):
  ^false1(%arg1: i32, %arg2: i32):
    spirv.Return
  }
}
