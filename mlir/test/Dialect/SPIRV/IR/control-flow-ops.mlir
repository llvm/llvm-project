// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.Branch
//===----------------------------------------------------------------------===//

func.func @branch() -> () {
  // CHECK: spirv.Branch ^bb1
  spirv.Branch ^next
^next:
  spirv.Return
}

// -----

func.func @branch_argument() -> () {
  %zero = spirv.Constant 0 : i32
  // CHECK: spirv.Branch ^bb1(%{{.*}}, %{{.*}} : i32, i32)
  spirv.Branch ^next(%zero, %zero: i32, i32)
^next(%arg0: i32, %arg1: i32):
  spirv.Return
}

// -----

func.func @missing_accessor() -> () {
  // expected-error @+1 {{expected block name}}
  spirv.Branch
}

// -----

func.func @wrong_accessor_count() -> () {
  %true = spirv.Constant true
  // expected-error @+1 {{requires 1 successor but found 2}}
  "spirv.Branch"()[^one, ^two] : () -> ()
^one:
  spirv.Return
^two:
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.BranchConditional
//===----------------------------------------------------------------------===//

func.func @cond_branch() -> () {
  %true = spirv.Constant true
  // CHECK: spirv.BranchConditional %{{.*}}, ^bb1, ^bb2
  spirv.BranchConditional %true, ^one, ^two
// CHECK: ^bb1
^one:
  spirv.Return
// CHECK: ^bb2
^two:
  spirv.Return
}

// -----

func.func @cond_branch_argument() -> () {
  %true = spirv.Constant true
  %zero = spirv.Constant 0 : i32
  // CHECK: spirv.BranchConditional %{{.*}}, ^bb1(%{{.*}}, %{{.*}} : i32, i32), ^bb2
  spirv.BranchConditional %true, ^true1(%zero, %zero: i32, i32), ^false1
^true1(%arg0: i32, %arg1: i32):
  // CHECK: spirv.BranchConditional %{{.*}}, ^bb3, ^bb4(%{{.*}}, %{{.*}} : i32, i32)
  spirv.BranchConditional %true, ^true2, ^false2(%zero, %zero: i32, i32)
^false1:
  spirv.Return
^true2:
  spirv.Return
^false2(%arg3: i32, %arg4: i32):
  spirv.Return
}

// -----

func.func @cond_branch_with_weights() -> () {
  %true = spirv.Constant true
  // CHECK: spirv.BranchConditional %{{.*}} [5, 10]
  spirv.BranchConditional %true [5, 10], ^one, ^two
^one:
  spirv.Return
^two:
  spirv.Return
}

// -----

func.func @missing_condition() -> () {
  // expected-error @+1 {{expected SSA operand}}
  spirv.BranchConditional ^one, ^two
^one:
  spirv.Return
^two:
  spirv.Return
}

// -----

func.func @wrong_condition_type() -> () {
  // expected-note @+1 {{prior use here}}
  %zero = spirv.Constant 0 : i32
  // expected-error @+1 {{use of value '%zero' expects different type than prior uses: 'i1' vs 'i32'}}
  spirv.BranchConditional %zero, ^one, ^two
^one:
  spirv.Return
^two:
  spirv.Return
}

// -----

func.func @wrong_accessor_count() -> () {
  %true = spirv.Constant true
  // expected-error @+1 {{requires 2 successors but found 1}}
  "spirv.BranchConditional"(%true)[^one] {operandSegmentSizes = array<i32: 1, 0, 0>} : (i1) -> ()
^one:
  spirv.Return
^two:
  spirv.Return
}

// -----

func.func @wrong_number_of_weights() -> () {
  %true = spirv.Constant true
  // expected-error @+1 {{must have exactly two branch weights}}
  "spirv.BranchConditional"(%true)[^one, ^two] {branch_weights = [1 : i32, 2 : i32, 3 : i32],
                                              operandSegmentSizes = array<i32: 1, 0, 0>} : (i1) -> ()
^one:
  spirv.Return
^two:
  spirv.Return
}

// -----

func.func @weights_cannot_both_be_zero() -> () {
  %true = spirv.Constant true
  // expected-error @+1 {{branch weights cannot both be zero}}
  spirv.BranchConditional %true [0, 0], ^one, ^two
^one:
  spirv.Return
^two:
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.FunctionCall
//===----------------------------------------------------------------------===//

spirv.module Logical GLSL450 {
  spirv.func @fmain(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>, %arg2 : i32) -> i32 "None" {
    // CHECK: {{%.*}} = spirv.FunctionCall @f_0({{%.*}}, {{%.*}}) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %0 = spirv.FunctionCall @f_0(%arg0, %arg1) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    // CHECK: spirv.FunctionCall @f_1({{%.*}}, {{%.*}}) : (vector<4xf32>, vector<4xf32>) -> ()
    spirv.FunctionCall @f_1(%0, %arg1) : (vector<4xf32>, vector<4xf32>) ->  ()
    // CHECK: spirv.FunctionCall @f_2() : () -> ()
    spirv.FunctionCall @f_2() : () -> ()
    // CHECK: {{%.*}} = spirv.FunctionCall @f_3({{%.*}}) : (i32) -> i32
    %1 = spirv.FunctionCall @f_3(%arg2) : (i32) -> i32
    spirv.ReturnValue %1 : i32
  }

  spirv.func @f_0(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> (vector<4xf32>) "None" {
    spirv.ReturnValue %arg0 : vector<4xf32>
  }

  spirv.func @f_1(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> () "None" {
    spirv.Return
  }

  spirv.func @f_2() -> () "None" {
    spirv.Return
  }

  spirv.func @f_3(%arg0 : i32) -> (i32) "None" {
    spirv.ReturnValue %arg0 : i32
  }
}

// -----

// Allow calling functions in other module-like ops
spirv.func @callee() "None" {
  spirv.Return
}

func.func @caller() {
  // CHECK: spirv.FunctionCall
  spirv.FunctionCall @callee() : () -> ()
  spirv.Return
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @f_invalid_result_type(%arg0 : i32, %arg1 : i32) -> () "None" {
    // expected-error @+1 {{result group starting at #0 requires 0 or 1 element, but found 2}}
    %0:2 = spirv.FunctionCall @f_invalid_result_type(%arg0, %arg1) : (i32, i32) -> (i32, i32)
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @f_result_type_mismatch(%arg0 : i32, %arg1 : i32) -> () "None" {
    // expected-error @+1 {{has incorrect number of results has for callee: expected 0, but provided 1}}
    %1 = spirv.FunctionCall @f_result_type_mismatch(%arg0, %arg0) : (i32, i32) -> (i32)
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @f_type_mismatch(%arg0 : i32, %arg1 : i32) -> () "None" {
    // expected-error @+1 {{has incorrect number of operands for callee: expected 2, but provided 1}}
    spirv.FunctionCall @f_type_mismatch(%arg0) : (i32) -> ()
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @f_type_mismatch(%arg0 : i32, %arg1 : i32) -> () "None" {
    %0 = spirv.Constant 2.0 : f32
    // expected-error @+1 {{operand type mismatch: expected operand type 'i32', but provided 'f32' for operand number 1}}
    spirv.FunctionCall @f_type_mismatch(%arg0, %0) : (i32, f32) -> ()
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @f_type_mismatch(%arg0 : i32, %arg1 : i32) -> i32 "None" {
    %cst = spirv.Constant 0: i32
    // expected-error @+1 {{result type mismatch: expected 'i32', but provided 'f32'}}
    %0 = spirv.FunctionCall @f_type_mismatch(%arg0, %arg0) : (i32, i32) -> f32
    spirv.ReturnValue %cst: i32
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @f_foo(%arg0 : i32, %arg1 : i32) -> i32 "None" {
    // expected-error @+1 {{op callee function 'f_undefined' not found in nearest symbol table}}
    %0 = spirv.FunctionCall @f_undefined(%arg0, %arg0) : (i32, i32) -> i32
    spirv.ReturnValue %0: i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.mlir.loop
//===----------------------------------------------------------------------===//

// for (int i = 0; i < count; ++i) {}
func.func @loop(%count : i32) -> () {
  %zero = spirv.Constant 0: i32
  %one = spirv.Constant 1: i32
  %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

  // CHECK: spirv.mlir.loop {
  spirv.mlir.loop {
    // CHECK-NEXT: spirv.Branch ^bb1
    spirv.Branch ^header

  // CHECK-NEXT: ^bb1:
  ^header:
    %val0 = spirv.Load "Function" %var : i32
    %cmp = spirv.SLessThan %val0, %count : i32
    // CHECK: spirv.BranchConditional %{{.*}}, ^bb2, ^bb4
    spirv.BranchConditional %cmp, ^body, ^merge

  // CHECK-NEXT: ^bb2:
  ^body:
    // Do nothing
    // CHECK-NEXT: spirv.Branch ^bb3
    spirv.Branch ^continue

  // CHECK-NEXT: ^bb3:
  ^continue:
    %val1 = spirv.Load "Function" %var : i32
    %add = spirv.IAdd %val1, %one : i32
    spirv.Store "Function" %var, %add : i32
    // CHECK: spirv.Branch ^bb1
    spirv.Branch ^header

  // CHECK-NEXT: ^bb4:
  ^merge:
    spirv.mlir.merge
  }
  return
}

// -----

// CHECK-LABEL: @empty_region
func.func @empty_region() -> () {
  // CHECK: spirv.mlir.loop
  spirv.mlir.loop {
  }
  return
}

// -----

// CHECK-LABEL: @loop_with_control
func.func @loop_with_control() -> () {
  // CHECK: spirv.mlir.loop control(Unroll)
  spirv.mlir.loop control(Unroll) {
  }
  return
}

// -----

func.func @wrong_merge_block() -> () {
  // expected-error @+1 {{last block must be the merge block with only one 'spirv.mlir.merge' op}}
  spirv.mlir.loop {
    spirv.Return
  }
  return
}

// -----

func.func @missing_entry_block() -> () {
  // expected-error @+1 {{must have an entry block branching to the loop header block}}
  spirv.mlir.loop {
    spirv.mlir.merge
  }
  return
}

// -----

func.func @missing_header_block() -> () {
  // expected-error @+1 {{must have a loop header block branched from the entry block}}
  spirv.mlir.loop {
  ^entry:
    spirv.Branch ^merge
  ^merge:
    spirv.mlir.merge
  }
  return
}

// -----

func.func @entry_should_branch_to_header() -> () {
  // expected-error @+1 {{entry block must only have one 'spirv.Branch' op to the second block}}
  spirv.mlir.loop {
  ^entry:
    spirv.Branch ^merge
  ^header:
    spirv.Branch ^merge
  ^merge:
    spirv.mlir.merge
  }
  return
}

// -----

func.func @missing_continue_block() -> () {
  // expected-error @+1 {{requires a loop continue block branching to the loop header block}}
  spirv.mlir.loop {
  ^entry:
    spirv.Branch ^header
  ^header:
    spirv.Branch ^merge
  ^merge:
    spirv.mlir.merge
  }
  return
}

// -----

func.func @continue_should_branch_to_header() -> () {
  // expected-error @+1 {{second to last block must be the loop continue block that branches to the loop header block}}
  spirv.mlir.loop {
  ^entry:
    spirv.Branch ^header
  ^header:
    spirv.Branch ^continue
  ^continue:
    spirv.Branch ^merge
  ^merge:
    spirv.mlir.merge
  }
  return
}

// -----

func.func @only_entry_and_continue_branch_to_header() -> () {
  // expected-error @+1 {{can only have the entry and loop continue block branching to the loop header block}}
  spirv.mlir.loop {
  ^entry:
    spirv.Branch ^header
  ^header:
    spirv.Branch ^cont1
  ^cont1:
    spirv.Branch ^header
  ^cont2:
    spirv.Branch ^header
  ^merge:
    spirv.mlir.merge
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.mlir.merge
//===----------------------------------------------------------------------===//

func.func @merge() -> () {
  // expected-error @+1 {{expected parent op to be 'spirv.mlir.selection' or 'spirv.mlir.loop'}}
  spirv.mlir.merge
}

// -----

func.func @only_allowed_in_last_block(%cond : i1) -> () {
  %zero = spirv.Constant 0: i32
  %one = spirv.Constant 1: i32
  %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^merge

  ^then:
    spirv.Store "Function" %var, %one : i32
    // expected-error @+1 {{can only be used in the last block of 'spirv.mlir.selection' or 'spirv.mlir.loop'}}
    spirv.mlir.merge

  ^merge:
    spirv.mlir.merge
  }

  spirv.Return
}

// -----

func.func @only_allowed_in_last_block() -> () {
  %true = spirv.Constant true
  spirv.mlir.loop {
    spirv.Branch ^header
  ^header:
    spirv.BranchConditional %true, ^body, ^merge
  ^body:
    // expected-error @+1 {{can only be used in the last block of 'spirv.mlir.selection' or 'spirv.mlir.loop'}}
    spirv.mlir.merge
  ^continue:
    spirv.Branch ^header
  ^merge:
    spirv.mlir.merge
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Return
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @in_selection
func.func @in_selection(%cond : i1) -> () {
  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^merge
  ^then:
    // CHECK: spirv.Return
    spirv.Return
  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}

// CHECK-LABEL: func @in_loop
func.func @in_loop(%cond : i1) -> () {
  spirv.mlir.loop {
    spirv.Branch ^header
  ^header:
    spirv.BranchConditional %cond, ^body, ^merge
  ^body:
    // CHECK: spirv.Return
    spirv.Return
  ^continue:
    spirv.Branch ^header
  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}

// CHECK-LABEL: in_other_func_like_op
func.func @in_other_func_like_op() {
  // CHECK: spirv.Return
  spirv.Return
}

// -----

"foo.function"() ({
  // expected-error @+1 {{op must appear in a function-like op's block}}
  spirv.Return
})  : () -> ()

// -----

// Return mismatches function signature
spirv.module Logical GLSL450 {
  spirv.func @work() -> (i32) "None" {
    // expected-error @+1 {{cannot be used in functions returning value}}
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @in_nested_region(%cond: i1) -> (i32) "None" {
    spirv.mlir.selection {
      spirv.BranchConditional %cond, ^then, ^merge
    ^then:
      // expected-error @+1 {{cannot be used in functions returning value}}
      spirv.Return
    ^merge:
      spirv.mlir.merge
    }

    %zero = spirv.Constant 0: i32
    spirv.ReturnValue %zero: i32
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ReturnValue
//===----------------------------------------------------------------------===//

func.func @ret_val() -> (i32) {
  %0 = spirv.Constant 42 : i32
  // CHECK: spirv.ReturnValue %{{.*}} : i32
  spirv.ReturnValue %0 : i32
}

// CHECK-LABEL: func @in_selection
func.func @in_selection(%cond : i1) -> (i32) {
  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^merge
  ^then:
    %zero = spirv.Constant 0 : i32
    // CHECK: spirv.ReturnValue
    spirv.ReturnValue %zero : i32
  ^merge:
    spirv.mlir.merge
  }
  %one = spirv.Constant 1 : i32
  spirv.ReturnValue %one : i32
}

// CHECK-LABEL: func @in_loop
func.func @in_loop(%cond : i1) -> (i32) {
  spirv.mlir.loop {
    spirv.Branch ^header
  ^header:
    spirv.BranchConditional %cond, ^body, ^merge
  ^body:
    %zero = spirv.Constant 0 : i32
    // CHECK: spirv.ReturnValue
    spirv.ReturnValue %zero : i32
  ^continue:
    spirv.Branch ^header
  ^merge:
    spirv.mlir.merge
  }
  %one = spirv.Constant 1 : i32
  spirv.ReturnValue %one : i32
}

// CHECK-LABEL: in_other_func_like_op
func.func @in_other_func_like_op(%arg: i32) -> i32 {
  // CHECK: spirv.ReturnValue
  spirv.ReturnValue %arg: i32
}

// -----

"foo.function"() ({
  %0 = spirv.Constant true
  // expected-error @+1 {{op must appear in a function-like op's block}}
  spirv.ReturnValue %0 : i1
})  : () -> ()

// -----

spirv.module Logical GLSL450 {
  spirv.func @value_count_mismatch() -> () "None" {
    %0 = spirv.Constant 42 : i32
    // expected-error @+1 {{op returns 1 value but enclosing function requires 0 results}}
    spirv.ReturnValue %0 : i32
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @value_type_mismatch() -> (f32) "None" {
    %0 = spirv.Constant 42 : i32
    // expected-error @+1 {{return value's type ('i32') mismatch with function's result type ('f32')}}
    spirv.ReturnValue %0 : i32
  }
}

// -----

spirv.module Logical GLSL450 {
  spirv.func @in_nested_region(%cond: i1) -> () "None" {
    spirv.mlir.selection {
      spirv.BranchConditional %cond, ^then, ^merge
    ^then:
      %cst = spirv.Constant 0: i32
      // expected-error @+1 {{op returns 1 value but enclosing function requires 0 results}}
      spirv.ReturnValue %cst: i32
    ^merge:
      spirv.mlir.merge
    }

    spirv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spirv.mlir.selection
//===----------------------------------------------------------------------===//

func.func @selection(%cond: i1) -> () {
  %zero = spirv.Constant 0: i32
  %one = spirv.Constant 1: i32
  %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

  // CHECK: spirv.mlir.selection {
  spirv.mlir.selection {
    // CHECK-NEXT: spirv.BranchConditional %{{.*}}, ^bb1, ^bb2
    spirv.BranchConditional %cond, ^then, ^merge

  // CHECK: ^bb1
  ^then:
    spirv.Store "Function" %var, %one : i32
    // CHECK: spirv.Branch ^bb2
    spirv.Branch ^merge

  // CHECK: ^bb2
  ^merge:
    // CHECK-NEXT: spirv.mlir.merge
    spirv.mlir.merge
  }

  spirv.Return
}

// -----

func.func @selection(%cond: i1) -> () {
  %zero = spirv.Constant 0: i32
  %one = spirv.Constant 1: i32
  %two = spirv.Constant 2: i32
  %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

  // CHECK: spirv.mlir.selection {
  spirv.mlir.selection {
    // CHECK-NEXT: spirv.BranchConditional %{{.*}}, ^bb1, ^bb2
    spirv.BranchConditional %cond, ^then, ^else

  // CHECK: ^bb1
  ^then:
    spirv.Store "Function" %var, %one : i32
    // CHECK: spirv.Branch ^bb3
    spirv.Branch ^merge

  // CHECK: ^bb2
  ^else:
    spirv.Store "Function" %var, %two : i32
    // CHECK: spirv.Branch ^bb3
    spirv.Branch ^merge

  // CHECK: ^bb3
  ^merge:
    // CHECK-NEXT: spirv.mlir.merge
    spirv.mlir.merge
  }

  spirv.Return
}

// -----

// CHECK-LABEL: @empty_region
func.func @empty_region() -> () {
  // CHECK: spirv.mlir.selection
  spirv.mlir.selection {
  }
  return
}

// -----

// CHECK-LABEL: @selection_with_control
func.func @selection_with_control() -> () {
  // CHECK: spirv.mlir.selection control(Flatten)
  spirv.mlir.selection control(Flatten) {
  }
  return
}

// -----

func.func @wrong_merge_block() -> () {
  // expected-error @+1 {{last block must be the merge block with only one 'spirv.mlir.merge' op}}
  spirv.mlir.selection {
    spirv.Return
  }
  return
}

// -----

func.func @missing_entry_block() -> () {
  // expected-error @+1 {{must have a selection header block}}
  spirv.mlir.selection {
    spirv.mlir.merge
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Unreachable
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unreachable_no_pred
func.func @unreachable_no_pred() {
    spirv.Return

  ^next:
    // CHECK: spirv.Unreachable
    spirv.Unreachable
}

// CHECK-LABEL: func @unreachable_with_pred
func.func @unreachable_with_pred() {
    spirv.Return

  ^parent:
    spirv.Branch ^unreachable

  ^unreachable:
    // CHECK: spirv.Unreachable
    spirv.Unreachable
}

// -----

func.func @unreachable() {
  // expected-error @+1 {{cannot be used in reachable block}}
  spirv.Unreachable
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Kill
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @kill
func.func @kill() {
  // CHECK: spirv.Kill
  spirv.Kill
}
