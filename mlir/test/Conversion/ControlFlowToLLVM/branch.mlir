// RUN: mlir-opt %s -convert-cf-to-llvm -split-input-file | FileCheck %s

// Unstructured control flow is converted, but the enclosing op is not
// converted.

// CHECK-LABEL: func.func @cf_br(
//  CHECK-SAME:     %[[arg0:.*]]: index) -> index {
//       CHECK:   %[[cast0:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : index to i64
//       CHECK:   llvm.br ^[[bb1:.*]](%[[cast0]] : i64)
//       CHECK: ^[[bb1]](%[[arg1:.*]]: i64):
//       CHECK:   %[[cast1:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : i64 to index
//       CHECK:   return %[[cast1]] : index
//       CHECK: }
func.func @cf_br(%arg0: index) -> index {
  cf.br ^bb1(%arg0 : index)
^bb1(%arg1: index):
  return %arg1 : index
}

// -----

// func.func and func.return types match. No unrealized_conversion_cast is
// needed.

// CHECK-LABEL: func.func @cf_br_type_match(
//  CHECK-SAME:     %[[arg0:.*]]: i64) -> i64 {
//       CHECK:   llvm.br ^[[bb1:.*]](%[[arg0:.*]] : i64)
//       CHECK: ^[[bb1]](%[[arg1:.*]]: i64):
//       CHECK:   return %[[arg1]] : i64
//       CHECK: }
func.func @cf_br_type_match(%arg0: i64) -> i64 {
  cf.br ^bb1(%arg0 : i64)
^bb1(%arg1: i64):
  return %arg1 : i64
}

// -----

// Test case for cf.cond_br.

//   CHECK-LABEL: func.func @cf_cond_br
// CHECK-COUNT-2:   unrealized_conversion_cast {{.*}} : index to i64
//         CHECK:   llvm.cond_br %{{.*}}, ^{{.*}}(%{{.*}} : i64), ^{{.*}}(%{{.*}} : i64)
//         CHECK: ^{{.*}}(%{{.*}}: i64):
//         CHECK:   unrealized_conversion_cast {{.*}} : i64 to index
//         CHECK: ^{{.*}}(%{{.*}}: i64):
//         CHECK:   unrealized_conversion_cast {{.*}} : i64 to index
func.func @cf_cond_br(%cond: i1, %a: index, %b: index) -> index {
  cf.cond_br %cond, ^bb1(%a : index), ^bb2(%b : index)
^bb1(%arg1: index):
  return %arg1 : index
^bb2(%arg2: index):
  return %arg2 : index
}

// -----

// Unreachable block (and IR in general) is not converted during a dialect
// conversion.

// CHECK-LABEL: func.func @unreachable_block()
//       CHECK:   return
//       CHECK: ^[[bb1:.*]](%[[arg0:.*]]: index):
//       CHECK:   cf.br ^[[bb1]](%[[arg0]] : index)
func.func @unreachable_block() {
  return
^bb1(%arg0: index):
  cf.br ^bb1(%arg0 : index)
}

// -----

// Test case for cf.cond_br with weights.

// CHECK-LABEL:   func.func @cf_cond_br_with_weights(
func.func @cf_cond_br_with_weights(%cond: i1, %a: index, %b: index) -> index {
// CHECK:           llvm.cond_br %{{.*}} weights([90, 10]), ^bb1(%{{.*}} : i64), ^bb2(%{{.*}} : i64)
  cf.cond_br %cond, ^bb1(%a : index), ^bb2(%b : index) {branch_weights = array<i32: 90, 10>}
^bb1(%arg1: index):
  return %arg1 : index
^bb2(%arg2: index):
  return %arg2 : index
}

// -----

// Verify that the pass respects the module's data layout when deriving the
// index type for block arguments.  When the module declares a 32-bit index
// via dlti.dl_spec, cf.br and cf.cond_br must pass index-typed block
// arguments as i32 instead of the default i64.

// 32-bit data layout: cf.br with index block argument -> i32 branch arg.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @cf_br_index_32bit(%arg0: index) -> index {
  cf.br ^bb1(%arg0 : index)
^bb1(%a: index):
  return %a : index
}

}

// CHECK-LABEL: func @cf_br_index_32bit
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: llvm.br ^{{.*}}(%{{.*}} : i32)
// CHECK: ^{{.*}}(%{{.*}}: i32):

// -----

// 32-bit data layout: cf.cond_br with index block arguments -> i32 branch args.

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {

func.func @cf_cond_br_index_32bit(%cond: i1, %a: index, %b: index) -> index {
  cf.cond_br %cond, ^bb1(%a : index), ^bb2(%b : index)
^bb1(%x: index):
  return %x : index
^bb2(%y: index):
  return %y : index
}

}

// CHECK-LABEL: func @cf_cond_br_index_32bit
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i32
// CHECK: llvm.cond_br %{{.*}}, ^{{.*}}(%{{.*}} : i32), ^{{.*}}(%{{.*}} : i32)

// -----

// Without dlti.dl_spec the default index width (i64) is preserved.

module {

func.func @cf_br_index_default(%arg0: index) -> index {
  cf.br ^bb1(%arg0 : index)
^bb1(%a: index):
  return %a : index
}

}

// CHECK-LABEL: func @cf_br_index_default
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i64
// CHECK: llvm.br ^{{.*}}(%{{.*}} : i64)
// CHECK: ^{{.*}}(%{{.*}}: i64):
