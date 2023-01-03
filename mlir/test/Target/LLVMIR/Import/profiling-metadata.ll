; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @cond_br
define i64 @cond_br(i1 %arg1, i64 %arg2) {
entry:
  ; CHECK: llvm.cond_br
  ; CHECK-SAME: weights(dense<[0, 3]> : vector<2xi32>)
  br i1 %arg1, label %bb1, label %bb2, !prof !0
bb1:
  ret i64 %arg2
bb2:
  ret i64 %arg2
}

!0 = !{!"branch_weights", i32 0, i32 3}

; // -----

; CHECK-LABEL: @simple_switch(
define i32 @simple_switch(i32 %arg1) {
  ; CHECK: llvm.switch
  ; CHECK: {branch_weights = dense<[42, 3, 5]> : vector<3xi32>}
  switch i32 %arg1, label %bbd [
    i32 0, label %bb1
    i32 9, label %bb2
  ], !prof !0
bb1:
  ret i32 %arg1
bb2:
  ret i32 %arg1
bbd:
  ret i32 %arg1
}

!0 = !{!"branch_weights", i32 42, i32 3, i32 5}
