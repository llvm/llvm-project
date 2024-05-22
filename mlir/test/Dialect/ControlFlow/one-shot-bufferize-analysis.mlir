// RUN: mlir-opt -one-shot-bufferize="test-analysis-only dump-alias-sets bufferize-function-boundaries" -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @single_branch(
//  CHECK-SAME:     {__bbarg_alias_set_attr__ = [{{\[}}[{{\[}}"%[[arg1:.*]]", "%[[t:.*]]"]], [{{\[}}"%[[arg1]]", "%[[t]]"]]]]}
func.func @single_branch(%t: tensor<5xf32>) -> tensor<5xf32> {
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  cf.br ^bb1(%t : tensor<5xf32>)
// CHECK: ^{{.*}}(%[[arg1]]: tensor<5xf32>)
^bb1(%arg1 : tensor<5xf32>):
  func.return %arg1 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func @diamond_branch(
//  CHECK-SAME:     %{{.*}}: i1, %[[t0:.*]]: tensor<5xf32> {{.*}}, %[[t1:.*]]: tensor<5xf32> {{.*}}) -> tensor<5xf32>
//  CHECK-SAME:     {__bbarg_alias_set_attr__ = [{{\[}}[{{\[}}"%[[arg1:.*]]", "%[[arg3:.*]]", "%[[arg2:.*]]", "%[[t0]]", "%[[t1]]"], [
func.func @diamond_branch(%c: i1, %t0: tensor<5xf32>, %t1: tensor<5xf32>) -> tensor<5xf32> {
// CHECK: cf.cond_br
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
  cf.cond_br %c, ^bb1(%t0 : tensor<5xf32>), ^bb2(%t1 : tensor<5xf32>)
// CHECK: ^{{.*}}(%[[arg1]]: tensor<5xf32>):
^bb3(%arg1 : tensor<5xf32>):
  func.return %arg1 : tensor<5xf32>
// CHECK: ^{{.*}}(%[[arg2]]: tensor<5xf32>):
^bb1(%arg2 : tensor<5xf32>):
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  cf.br ^bb3(%arg2 : tensor<5xf32>)
// CHECK: ^{{.*}}(%[[arg3]]: tensor<5xf32>):
^bb2(%arg3 : tensor<5xf32>):
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  cf.br ^bb3(%arg3 : tensor<5xf32>)
}

// -----

// CHECK-LABEL: func @looping_branches(
//  CHECK-SAME:     {__bbarg_alias_set_attr__ = [{{\[}}[], [{{\[}}"%[[arg2:.*]]", "%[[arg1:.*]]", "%[[inserted:.*]]", "%[[empty:.*]]"]], [
func.func @looping_branches() -> tensor<5xf32> {
// CHECK: %[[empty]] = tensor.empty()
  %0 = tensor.empty() : tensor<5xf32>
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  cf.br ^bb1(%0: tensor<5xf32>)
// CHECK: ^{{.*}}(%[[arg1]]: tensor<5xf32>):
^bb1(%arg1: tensor<5xf32>):
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: %[[inserted]] = tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %arg1[%pos] : tensor<5xf32>
  %cond = "test.qux"() : () -> (i1)
// CHECK: cf.cond_br
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
  cf.cond_br %cond, ^bb1(%inserted: tensor<5xf32>), ^bb2(%inserted: tensor<5xf32>)
^bb2(%arg2: tensor<5xf32>):
  func.return %arg2 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func @looping_branches_with_conflict(
func.func @looping_branches_with_conflict(%f: f32) -> tensor<5xf32> {
  %0 = tensor.empty() : tensor<5xf32>
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  cf.br ^bb1(%filled: tensor<5xf32>)
^bb2(%arg2: tensor<5xf32>):
  %pos2 = "test.foo"() : () -> (index)
  // One OpOperand cannot bufferize in-place because an "old" value is read.
  %element = tensor.extract %filled[%pos2] : tensor<5xf32>
  func.return %arg2 : tensor<5xf32>
^bb1(%arg1: tensor<5xf32>):
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %arg1[%pos] : tensor<5xf32>
  %cond = "test.qux"() : () -> (i1)
// CHECK: cf.cond_br
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
  cf.cond_br %cond, ^bb1(%inserted: tensor<5xf32>), ^bb2(%inserted: tensor<5xf32>)
}

// -----

// CHECK-LABEL: func @looping_branches_outside_def(
func.func @looping_branches_outside_def(%f: f32) {
// CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
// CHECK: %[[fill:.*]] = linalg.fill
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"], __opresult_alias_set_attr__ = [{{\[}}"%[[fill]]", "%[[alloc]]"]]}
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
  cf.br ^bb1
^bb1:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %filled[%pos] : tensor<5xf32>
  %pos2 = "test.foo"() : () -> (index)
  %read = tensor.extract %inserted[%pos2] : tensor<5xf32>
  %cond = "test.qux"(%read) : (f32) -> (i1)
  cf.cond_br %cond, ^bb1, ^bb2
^bb2:
  func.return
}

// -----

// CHECK-LABEL: func @looping_branches_outside_def2(
func.func @looping_branches_outside_def2(%f: f32) {
// CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
// CHECK: %[[fill:.*]] = linalg.fill
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"], __opresult_alias_set_attr__ = [{{\[}}"%[[arg0:.*]]", "%[[fill]]", "%[[alloc]]"]]}
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
// CHECK: cf.br {{.*}}(%[[fill]] : tensor<5xf32>)
// CHECK-SAME: __inplace_operands_attr__ = ["true"]
  cf.br ^bb1(%filled: tensor<5xf32>)
// CHECK: ^{{.*}}(%[[arg0]]: tensor<5xf32>):
^bb1(%arg0: tensor<5xf32>):
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %arg0[%pos] : tensor<5xf32>
  %pos2 = "test.foo"() : () -> (index)
  %read = tensor.extract %inserted[%pos2] : tensor<5xf32>
  %cond = "test.qux"(%read) : (f32) -> (i1)
// CHECK: cf.cond_br
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true"]
  cf.cond_br %cond, ^bb1(%arg0: tensor<5xf32>), ^bb2
^bb2:
  func.return
}

// -----

// CHECK-LABEL: func @looping_branches_outside_def3(
func.func @looping_branches_outside_def3(%f: f32) {
// CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
// CHECK: %[[fill:.*]] = linalg.fill
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"], __opresult_alias_set_attr__ = [{{\[}}"%[[arg0:.*]]", "%[[fill]]", "%[[alloc]]"]]}
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
// CHECK: cf.br {{.*}}(%[[fill]] : tensor<5xf32>)
// CHECK-SAME: __inplace_operands_attr__ = ["true"]
  cf.br ^bb1(%filled: tensor<5xf32>)
// CHECK: ^{{.*}}(%[[arg0]]: tensor<5xf32>):
^bb1(%arg0: tensor<5xf32>):
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %arg0[%pos] : tensor<5xf32>
  %pos2 = "test.foo"() : () -> (index)
  %read = tensor.extract %inserted[%pos2] : tensor<5xf32>
  %cond = "test.qux"(%read) : (f32) -> (i1)
// CHECK: cf.cond_br
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true"]
  cf.cond_br %cond, ^bb1(%filled: tensor<5xf32>), ^bb2
^bb2:
  func.return
}

// -----

// CHECK-LABEL: func @looping_branches_sequence_outside_def(
func.func @looping_branches_sequence_outside_def(%f: f32) {
// CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
// CHECK: %[[fill:.*]] = linalg.fill
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"], __opresult_alias_set_attr__ = [{{\[}}"%[[fill]]", "%[[alloc]]"]]}
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
  cf.br ^bb1
^bb1:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %filled[%pos] : tensor<5xf32>
  cf.br ^bb2
^bb2:
  %pos2 = "test.foo"() : () -> (index)
  %read = tensor.extract %inserted[%pos2] : tensor<5xf32>
  %cond = "test.qux"(%read) : (f32) -> (i1)
  cf.cond_br %cond, ^bb1, ^bb3
^bb3:
  func.return
}

// -----

// CHECK-LABEL: func @looping_branches_sequence_inside_def(
func.func @looping_branches_sequence_inside_def(%f: f32) {
  cf.br ^bb1
^bb1:
// CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
// CHECK: %[[fill:.*]] = linalg.fill
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"], __opresult_alias_set_attr__ = [{{\[}}"%[[inserted:.*]]", "%[[fill]]", "%[[alloc]]"]]}
  %filled = linalg.fill ins(%f : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: %[[inserted]] = tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %filled[%pos] : tensor<5xf32>
  cf.br ^bb2
^bb2:
  %pos2 = "test.foo"() : () -> (index)
  %read = tensor.extract %inserted[%pos2] : tensor<5xf32>
  %cond = "test.qux"(%read) : (f32) -> (i1)
  cf.cond_br %cond, ^bb1, ^bb3
^bb3:
  func.return
}
