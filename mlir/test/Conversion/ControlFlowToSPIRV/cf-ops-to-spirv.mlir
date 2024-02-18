// RUN: mlir-opt --split-input-file --convert-cf-to-spirv --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// cf.br, cf.cond_br
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func @simple_loop
func.func @simple_loop(%begin: i32, %end: i32, %step: i32) {
// CHECK-NEXT:  spirv.Branch ^bb1
  cf.br ^bb1

// CHECK-NEXT: ^bb1:    // pred: ^bb0
// CHECK-NEXT:  spirv.Branch ^bb2({{.*}} : i32)
^bb1:   // pred: ^bb0
  cf.br ^bb2(%begin : i32)

// CHECK:      ^bb2({{.*}}: i32):       // 2 preds: ^bb1, ^bb3
// CHECK:        spirv.BranchConditional {{.*}}, ^bb3, ^bb4
^bb2(%0: i32):        // 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %0, %end : i32
  cf.cond_br %1, ^bb3, ^bb4

// CHECK:      ^bb3:    // pred: ^bb2
// CHECK:        spirv.Branch ^bb2({{.*}} : i32)
^bb3:   // pred: ^bb2
  %2 = arith.addi %0, %step : i32
  cf.br ^bb2(%2 : i32)

// CHECK:      ^bb4:    // pred: ^bb2
^bb4:   // pred: ^bb2
  return
}

}

// -----

// Handle blocks whose arguments require type conversion.

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%arg0: index) {
  %c3 = arith.constant 1 : index
// CHECK:  spirv.Branch ^bb1({{.*}} : i32)
  cf.br ^bb1(%arg0 : index)
// CHECK:      ^bb1({{.*}}: i32):       // 2 preds: ^bb0, ^bb2
^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
  %1 = arith.cmpi slt, %0, %c3 : index
// CHECK:        spirv.BranchConditional {{.*}}, ^bb2, ^bb3
  cf.cond_br %1, ^bb2, ^bb3
^bb2:  // pred: ^bb1
// CHECK:  spirv.Branch ^bb1({{.*}} : i32)
  cf.br ^bb1(%c3 : index)
^bb3:  // pred: ^bb1
  return
}
