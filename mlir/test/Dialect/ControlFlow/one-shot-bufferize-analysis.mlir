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

// -----

// Sequential loops; loop 2 forwards loop 1's bbarg. Inserts are inplace.
// CHECK-LABEL: func @sequential_loops_forward_bbarg(
func.func @sequential_loops_forward_bbarg() -> tensor<5xf32> {
  %init = tensor.empty() : tensor<5xf32>
  %enter = "test.foo"() : () -> (i1)
// CHECK: cf.cond_br
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
  cf.cond_br %enter, ^bb1(%init : tensor<5xf32>), ^bb1(%init : tensor<5xf32>)

^bb1(%cache: tensor<5xf32>):
  %c1 = "test.foo"() : () -> (i1)
  cf.cond_br %c1, ^bb3, ^bb2

^bb2:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %mid = tensor.insert %val into %cache[%pos] : tensor<5xf32>
  %pos2 = "test.foo"() : () -> (index)
  %val2 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %updated1 = tensor.insert %val2 into %mid[%pos2] : tensor<5xf32>
// CHECK: cf.cond_br
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
  cf.cond_br %enter, ^bb1(%updated1 : tensor<5xf32>), ^bb1(%updated1 : tensor<5xf32>)

^bb3:
// CHECK: cf.cond_br
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
  cf.cond_br %enter, ^bb4(%cache : tensor<5xf32>), ^bb4(%cache : tensor<5xf32>)

^bb4(%cache2: tensor<5xf32>):
  %c2 = "test.foo"() : () -> (i1)
  cf.cond_br %c2, ^bb6, ^bb5

^bb5:
  %pos3 = "test.foo"() : () -> (index)
  %val3 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %updated2 = tensor.insert %val3 into %cache2[%pos3] : tensor<5xf32>
// CHECK: cf.cond_br
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
  cf.cond_br %enter, ^bb4(%updated2 : tensor<5xf32>), ^bb4(%updated2 : tensor<5xf32>)

^bb6:
  func.return %cache2 : tensor<5xf32>
}

// -----

// Diamond: read vs write of an OpResult dest. Insert is inplace.
// CHECK-LABEL: func @diamond_read_write(
func.func @diamond_read_write(%c: i1, %f: f32) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c, ^bb2, ^bb5
^bb2:
  %pos = "test.foo"() : () -> (index)
// CHECK: tensor.extract
// CHECK-SAME: __inplace_operands_attr__ = ["true", "none"]
  %extracted = tensor.extract %t[%pos] : tensor<5xf32>
  "test.qux"(%extracted) : (f32) -> ()
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  cf.br ^bb6(%t : tensor<5xf32>)
^bb5:
  %pos2 = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %t[%pos2] : tensor<5xf32>
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  cf.br ^bb6(%inserted : tensor<5xf32>)
^bb6(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

// -----

// Reads in both arms, write after the join. Insert is inplace.
// CHECK-LABEL: func @read_then_write_after_join(
func.func @read_then_write_after_join(%c: i1, %f: f32) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  %pos = "test.foo"() : () -> (index)
  %e1 = tensor.extract %t[%pos] : tensor<5xf32>
  "test.qux"(%e1) : (f32) -> ()
  cf.br ^bb3
^bb2:
  %pos2 = "test.foo"() : () -> (index)
  %e2 = tensor.extract %t[%pos2] : tensor<5xf32>
  "test.qux"(%e2) : (f32) -> ()
  cf.br ^bb3
^bb3:
  %pos3 = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %t[%pos3] : tensor<5xf32>
  func.return %inserted : tensor<5xf32>
}

// -----

// Write in one arm, extract of the original after the join. Insert is out-of-place.
// CHECK-LABEL: func @write_then_read_after_join(
func.func @write_then_read_after_join(%c: i1, %f: f32) -> f32 {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %t[%pos] : tensor<5xf32>
  cf.br ^bb3(%inserted : tensor<5xf32>)
^bb2:
  cf.br ^bb3(%t : tensor<5xf32>)
^bb3(%r: tensor<5xf32>):
  %pos2 = "test.foo"() : () -> (index)
  %e = tensor.extract %t[%pos2] : tensor<5xf32>
  func.return %e : f32
}

// -----

// bb0 -> bb2|bb5, bb2 -> bb5|bb6. Read in bb2, write in bb5. Insert is inplace.
// CHECK-LABEL: func @read_predecessor_write_successor(
func.func @read_predecessor_write_successor(%c: i1, %c2: i1, %f: f32) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c, ^bb2, ^bb5
^bb2:
  %pos = "test.foo"() : () -> (index)
  %e = tensor.extract %t[%pos] : tensor<5xf32>
  "test.qux"(%e) : (f32) -> ()
  cf.cond_br %c2, ^bb5, ^bb6(%t : tensor<5xf32>)
^bb5:
  %pos2 = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %t[%pos2] : tensor<5xf32>
  cf.br ^bb6(%inserted : tensor<5xf32>)
^bb6(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

// -----

// Same CFG; write in bb2, extract of the original in bb5. Insert is out-of-place.
// CHECK-LABEL: func @write_predecessor_read_successor(
func.func @write_predecessor_read_successor(%c: i1, %c2: i1, %f: f32) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c, ^bb2, ^bb5
^bb2:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %t[%pos] : tensor<5xf32>
  cf.cond_br %c2, ^bb5, ^bb6(%inserted : tensor<5xf32>)
^bb5:
  %pos2 = "test.foo"() : () -> (index)
  %e = tensor.extract %t[%pos2] : tensor<5xf32>
  "test.qux"(%e) : (f32) -> ()
  cf.br ^bb6(%t : tensor<5xf32>)
^bb6(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

// -----

// Three exclusive arms: read, write, passthrough. Insert is inplace.
// CHECK-LABEL: func @three_way_exclusive_read_write(
func.func @three_way_exclusive_read_write(%c1: i1, %c2: i1, %f: f32) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c1, ^read, ^rest
^rest:
  cf.cond_br %c2, ^write, ^pass
^read:
  %pos = "test.foo"() : () -> (index)
  %e = tensor.extract %t[%pos] : tensor<5xf32>
  "test.qux"(%e) : (f32) -> ()
  cf.br ^join(%t : tensor<5xf32>)
^write:
  %pos2 = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %t[%pos2] : tensor<5xf32>
  cf.br ^join(%inserted : tensor<5xf32>)
^pass:
  cf.br ^join(%t : tensor<5xf32>)
^join(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

// -----

// Both diamond arms write the same dest. Both inserts are inplace.
// CHECK-LABEL: func @writes_in_both_diamond_arms(
func.func @writes_in_both_diamond_arms(%c: i1, %f: f32) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %i1 = tensor.insert %val into %t[%pos] : tensor<5xf32>
  cf.br ^bb3(%i1 : tensor<5xf32>)
^bb2:
  %pos2 = "test.foo"() : () -> (index)
  %val2 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %i2 = tensor.insert %val2 into %t[%pos2] : tensor<5xf32>
  cf.br ^bb3(%i2 : tensor<5xf32>)
^bb3(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

// -----

// Nested diamond: inner read vs outer write. Insert is inplace.
// CHECK-LABEL: func @nested_diamond_exclusive(
func.func @nested_diamond_exclusive(%c: i1, %c2: i1, %f: f32) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c, ^outer_read, ^outer_write
^outer_read:
  cf.cond_br %c2, ^inner_read, ^inner_pass
^inner_read:
  %pos = "test.foo"() : () -> (index)
  %e = tensor.extract %t[%pos] : tensor<5xf32>
  "test.qux"(%e) : (f32) -> ()
  cf.br ^join(%t : tensor<5xf32>)
^inner_pass:
  cf.br ^join(%t : tensor<5xf32>)
^outer_write:
  %pos2 = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %t[%pos2] : tensor<5xf32>
  cf.br ^join(%inserted : tensor<5xf32>)
^join(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

// -----

// Diamond inside a loop; dest defined outside. Insert is out-of-place.
// CHECK-LABEL: func @diamond_inside_loop_outside_def(
func.func @diamond_inside_loop_outside_def(%c: i1, %back: i1, %f: f32) -> tensor<5xf32> {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.br ^header
^header:
  cf.cond_br %c, ^read, ^write
^read:
  %pos = "test.foo"() : () -> (index)
  %e = tensor.extract %t[%pos] : tensor<5xf32>
  "test.qux"(%e) : (f32) -> ()
  cf.br ^join
^write:
  %pos2 = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %t[%pos2] : tensor<5xf32>
  cf.br ^join
^join:
  cf.cond_br %back, ^header, ^exit
^exit:
  func.return %t : tensor<5xf32>
}

// -----

// Diamond inside a loop; dest defined in the loop body. Insert is out-of-place.
// CHECK-LABEL: func @diamond_inside_loop_inside_def(
func.func @diamond_inside_loop_inside_def(%c: i1, %back: i1, %f: f32) -> tensor<5xf32> {
  cf.br ^header
^header:
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.cond_br %c, ^read, ^write
^read:
  %pos = "test.foo"() : () -> (index)
  %e = tensor.extract %t[%pos] : tensor<5xf32>
  "test.qux"(%e) : (f32) -> ()
  cf.br ^join(%t : tensor<5xf32>)
^write:
  %pos2 = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %t[%pos2] : tensor<5xf32>
  cf.br ^join(%inserted : tensor<5xf32>)
^join(%r: tensor<5xf32>):
  cf.cond_br %back, ^header, ^exit(%r : tensor<5xf32>)
^exit(%out: tensor<5xf32>):
  func.return %out : tensor<5xf32>
}

// -----

// Loop then diamond on the forwarded bbarg. Inserts are inplace.
// CHECK-LABEL: func @sequential_loop_then_diamond(
func.func @sequential_loop_then_diamond(%c: i1, %enter: i1) -> tensor<5xf32> {
  %init = tensor.empty() : tensor<5xf32>
  cf.cond_br %enter, ^loop(%init : tensor<5xf32>), ^loop(%init : tensor<5xf32>)
^loop(%cache: tensor<5xf32>):
  %c1 = "test.foo"() : () -> (i1)
  cf.cond_br %c1, ^after, ^body
^body:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %mid = tensor.insert %val into %cache[%pos] : tensor<5xf32>
  %pos2 = "test.foo"() : () -> (index)
  %val2 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %updated = tensor.insert %val2 into %mid[%pos2] : tensor<5xf32>
// CHECK: cf.cond_br
// CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
  cf.cond_br %enter, ^loop(%updated : tensor<5xf32>), ^loop(%updated : tensor<5xf32>)
^after:
  cf.cond_br %c, ^read(%cache : tensor<5xf32>), ^write(%cache : tensor<5xf32>)
^read(%r0: tensor<5xf32>):
  %pos3 = "test.foo"() : () -> (index)
  %e = tensor.extract %r0[%pos3] : tensor<5xf32>
  "test.qux"(%e) : (f32) -> ()
  cf.br ^join(%r0 : tensor<5xf32>)
^write(%r1: tensor<5xf32>):
  %pos4 = "test.foo"() : () -> (index)
  %val3 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val3 into %r1[%pos4] : tensor<5xf32>
  cf.br ^join(%inserted : tensor<5xf32>)
^join(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

// -----

// Sequential inner loops inside an outer loop. There is no actual RaW
// conflict: the only CFG path from the l2 insert to the later l1 insert is
// the outer-loop back-edge, which is a different outer iteration. The
// analysis is currently conservative with regards to that back-edge, so the
// first insert dest is out-of-place.
// CHECK-LABEL: func @outer_loop_sequential_inner_loops(
func.func @outer_loop_sequential_inner_loops(%again: i1) -> tensor<5xf32> {
  %init = tensor.empty() : tensor<5xf32>
  cf.br ^outer(%init : tensor<5xf32>)
^outer(%cache: tensor<5xf32>):
  cf.br ^l1(%cache : tensor<5xf32>)
^l1(%c1: tensor<5xf32>):
  %e1 = "test.foo"() : () -> (i1)
  cf.cond_br %e1, ^l2entry, ^l1body
^l1body:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
  // No actual conflict; conservative on the outer-loop back-edge.
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %mid = tensor.insert %val into %c1[%pos] : tensor<5xf32>
  %pos2 = "test.foo"() : () -> (index)
  %val2 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %upd1 = tensor.insert %val2 into %mid[%pos2] : tensor<5xf32>
  cf.br ^l1(%upd1 : tensor<5xf32>)
^l2entry:
  cf.br ^l2(%c1 : tensor<5xf32>)
^l2(%c2: tensor<5xf32>):
  %e2 = "test.foo"() : () -> (i1)
  cf.cond_br %e2, ^outer_end, ^l2body
^l2body:
  %pos3 = "test.foo"() : () -> (index)
  %val3 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %upd2 = tensor.insert %val3 into %c2[%pos3] : tensor<5xf32>
  cf.br ^l2(%upd2 : tensor<5xf32>)
^outer_end:
  cf.cond_br %again, ^outer(%c2 : tensor<5xf32>), ^exit
^exit:
  func.return %c2 : tensor<5xf32>
}

// -----

// Sequential inner loops inside an outer loop; dest defined outside. The
// outer backedge lets l1 write and l2 read the same outside value on
// different iterations, so the insert dest is out-of-place.
// CHECK-LABEL: func @outer_loop_seq_inners_outside_def(
func.func @outer_loop_seq_inners_outside_def(%c: i1, %back: i1, %f: f32) -> f32 {
  %empty = tensor.empty() : tensor<5xf32>
  %t = linalg.fill ins(%f : f32) outs(%empty : tensor<5xf32>) -> tensor<5xf32>
  cf.br ^outer
^outer:
  cf.br ^l1
^l1:
  %e1 = "test.foo"() : () -> (i1)
  cf.cond_br %e1, ^l2, ^l1body
^l1body:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %mid = tensor.insert %val into %t[%pos] : tensor<5xf32>
  cf.br ^l1
^l2:
  %e2 = "test.foo"() : () -> (i1)
  cf.cond_br %e2, ^outer_end, ^l2body
^l2body:
  %pos2 = "test.foo"() : () -> (index)
  %ex = tensor.extract %t[%pos2] : tensor<5xf32>
  "test.qux"(%ex) : (f32) -> ()
  cf.br ^l2
^outer_end:
  cf.cond_br %back, ^outer, ^exit
^exit:
  %pos3 = "test.foo"() : () -> (index)
  %r = tensor.extract %t[%pos3] : tensor<5xf32>
  func.return %r : f32
}

// -----

// l1 writes the header bbarg then extracts it. The extract must observe %c1
// before the insert, so the insert dest is out-of-place.
// CHECK-LABEL: func @l1_write_then_extract_bbarg(
func.func @l1_write_then_extract_bbarg(%again: i1) -> tensor<5xf32> {
  %init = tensor.empty() : tensor<5xf32>
  cf.br ^outer(%init : tensor<5xf32>)
^outer(%cache: tensor<5xf32>):
  cf.br ^l1(%cache : tensor<5xf32>)
^l1(%c1: tensor<5xf32>):
  %e1 = "test.foo"() : () -> (i1)
  cf.cond_br %e1, ^l2entry, ^l1body
^l1body:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %upd1 = tensor.insert %val into %c1[%pos] : tensor<5xf32>
  %posr = "test.foo"() : () -> (index)
  %old = tensor.extract %c1[%posr] : tensor<5xf32>
  "test.qux"(%old) : (f32) -> ()
  cf.br ^l1(%upd1 : tensor<5xf32>)
^l2entry:
  cf.br ^l2(%c1 : tensor<5xf32>)
^l2(%c2: tensor<5xf32>):
  %e2 = "test.foo"() : () -> (i1)
  cf.cond_br %e2, ^outer_end, ^l2body
^l2body:
  %pos3 = "test.foo"() : () -> (index)
  %val3 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %upd2 = tensor.insert %val3 into %c2[%pos3] : tensor<5xf32>
  cf.br ^l2(%upd2 : tensor<5xf32>)
^outer_end:
  cf.cond_br %again, ^outer(%c2 : tensor<5xf32>), ^exit
^exit:
  func.return %c2 : tensor<5xf32>
}

// -----

// Extra edge l2body -> l1body, not through the outer header. The tensor is
// copied on the branches into the inner loops.
// CHECK-LABEL: func @irreducible_l2_to_l1(
func.func @irreducible_l2_to_l1(%again: i1, %weird: i1) -> tensor<5xf32> {
  %init = tensor.empty() : tensor<5xf32>
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  cf.br ^outer(%init : tensor<5xf32>)
^outer(%cache: tensor<5xf32>):
// CHECK: cf.br
// CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  cf.br ^l1(%cache : tensor<5xf32>)
^l1(%c1: tensor<5xf32>):
  %e1 = "test.foo"() : () -> (i1)
  cf.cond_br %e1, ^l2entry, ^l1body
^l1body:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %mid = tensor.insert %val into %c1[%pos] : tensor<5xf32>
  %pos2 = "test.foo"() : () -> (index)
  %val2 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %upd1 = tensor.insert %val2 into %mid[%pos2] : tensor<5xf32>
  cf.br ^l1(%upd1 : tensor<5xf32>)
^l2entry:
  cf.br ^l2(%c1 : tensor<5xf32>)
^l2(%c2: tensor<5xf32>):
  %e2 = "test.foo"() : () -> (i1)
  cf.cond_br %e2, ^outer_end, ^l2body
^l2body:
  %pos3 = "test.foo"() : () -> (index)
  %val3 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %upd2 = tensor.insert %val3 into %c2[%pos3] : tensor<5xf32>
  cf.cond_br %weird, ^l1body, ^l2(%upd2 : tensor<5xf32>)
^outer_end:
  cf.cond_br %again, ^outer(%c2 : tensor<5xf32>), ^exit
^exit:
  func.return %c2 : tensor<5xf32>
}

// -----

// l1 writes the carried dest then extracts the outer bbarg. The extract must
// observe %cache before the first write, so that insert dest is out-of-place.
// The chained insert writes the copy and is inplace.
// CHECK-LABEL: func @l1_write_then_extract_carried(
func.func @l1_write_then_extract_carried(%again: i1) -> tensor<5xf32> {
  %init = tensor.empty() : tensor<5xf32>
  cf.br ^outer(%init : tensor<5xf32>)
^outer(%cache: tensor<5xf32>):
  cf.br ^l1(%cache : tensor<5xf32>)
^l1(%c1: tensor<5xf32>):
  %e1 = "test.foo"() : () -> (i1)
  cf.cond_br %e1, ^l2entry, ^l1body
^l1body:
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %mid = tensor.insert %val into %c1[%pos] : tensor<5xf32>
  %pos2 = "test.foo"() : () -> (index)
  %val2 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %upd1 = tensor.insert %val2 into %mid[%pos2] : tensor<5xf32>
  %posr = "test.foo"() : () -> (index)
  %old = tensor.extract %cache[%posr] : tensor<5xf32>
  "test.qux"(%old) : (f32) -> ()
  cf.br ^l1(%upd1 : tensor<5xf32>)
^l2entry:
  cf.br ^l2(%c1 : tensor<5xf32>)
^l2(%c2: tensor<5xf32>):
  %e2 = "test.foo"() : () -> (i1)
  cf.cond_br %e2, ^outer_end, ^l2body
^l2body:
  %pos3 = "test.foo"() : () -> (index)
  %val3 = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %upd2 = tensor.insert %val3 into %c2[%pos3] : tensor<5xf32>
  cf.br ^l2(%upd2 : tensor<5xf32>)
^outer_end:
  cf.cond_br %again, ^outer(%c2 : tensor<5xf32>), ^exit
^exit:
  func.return %c2 : tensor<5xf32>
}

// -----

// Mutually exclusive cases each take a tensor bbArg. Only one is live at
// runtime. That does not put every case block into the RaW extra-barrier set:
// findDefinitions stops at a bbArg, and sibling case bbArgs are not in scope
// together, so they cannot all be definitions of one read.
// Passing the original dest to the join after the insert is a RaW, so the
// insert dest is out-of-place.
// CHECK-LABEL: func @switch_write_pass_original(
func.func @switch_write_pass_original(%flag: i32) -> tensor<5xf32> {
  %init = tensor.empty() : tensor<5xf32>
  cf.switch %flag : i32, [
    default: ^c0(%init : tensor<5xf32>),
    1: ^c1(%init : tensor<5xf32>)
  ]
^c0(%a0: tensor<5xf32>):
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %inserted = tensor.insert %val into %a0[%pos] : tensor<5xf32>
  cf.br ^join(%a0 : tensor<5xf32>)
^c1(%a1: tensor<5xf32>):
  cf.br ^join(%a1 : tensor<5xf32>)
^join(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

// -----

// Same switch shape; the join is the insert result, so there is no RaW and
// the insert dest is inplace.
// CHECK-LABEL: func @switch_write_pass_result(
func.func @switch_write_pass_result(%flag: i32) -> tensor<5xf32> {
  %init = tensor.empty() : tensor<5xf32>
  cf.switch %flag : i32, [
    default: ^c0(%init : tensor<5xf32>),
    1: ^c1(%init : tensor<5xf32>)
  ]
^c0(%a0: tensor<5xf32>):
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: tensor.insert
// CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %inserted = tensor.insert %val into %a0[%pos] : tensor<5xf32>
  cf.br ^join(%inserted : tensor<5xf32>)
^c1(%a1: tensor<5xf32>):
  cf.br ^join(%a1 : tensor<5xf32>)
^join(%r: tensor<5xf32>):
  func.return %r : tensor<5xf32>
}

