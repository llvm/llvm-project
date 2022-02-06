// RUN: mlir-opt -split-input-file -convert-std-to-spirv -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// cf.br, cf.cond_br
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: func @simple_loop
func @simple_loop(index, index, index) {
^bb0(%begin : index, %end : index, %step : index):
// CHECK-NEXT:  spv.Branch ^bb1
  cf.br ^bb1

// CHECK-NEXT: ^bb1:    // pred: ^bb0
// CHECK-NEXT:  spv.Branch ^bb2({{.*}} : i32)
^bb1:   // pred: ^bb0
  cf.br ^bb2(%begin : index)

// CHECK:      ^bb2({{.*}}: i32):       // 2 preds: ^bb1, ^bb3
// CHECK-NEXT:  {{.*}} = spv.SLessThan {{.*}}, {{.*}} : i32
// CHECK-NEXT:  spv.BranchConditional {{.*}}, ^bb3, ^bb4
^bb2(%0: index):        // 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %0, %end : index
  cf.cond_br %1, ^bb3, ^bb4

// CHECK:      ^bb3:    // pred: ^bb2
// CHECK-NEXT:  {{.*}} = spv.IAdd {{.*}}, {{.*}} : i32
// CHECK-NEXT:  spv.Branch ^bb2({{.*}} : i32)
^bb3:   // pred: ^bb2
  %2 = arith.addi %0, %step : index
  cf.br ^bb2(%2 : index)

// CHECK:      ^bb4:    // pred: ^bb2
^bb4:   // pred: ^bb2
  return
}

}
