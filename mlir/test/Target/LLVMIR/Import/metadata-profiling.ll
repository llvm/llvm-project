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

; // -----

; CHECK: llvm.func @fn()
declare void @fn()

; CHECK-LABEL: @call_branch_weights
define void @call_branch_weights() {
  ; CHECK:  llvm.call @fn() {branch_weights = dense<42> : vector<1xi32>}
  call void @fn(), !prof !0
  ret void
}

!0 = !{!"branch_weights", i32 42}

; // -----

declare void @foo()
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: @invoke_branch_weights
define i32 @invoke_branch_weights() personality ptr @__gxx_personality_v0 {
  ; CHECK: llvm.invoke @foo() to ^bb2 unwind ^bb1 {branch_weights = dense<[42, 99]> : vector<2xi32>} : () -> ()
  invoke void @foo() to label %bb2 unwind label %bb1, !prof !0
bb1:
  %1 = landingpad { ptr, i32 } cleanup
  br label %bb2
bb2:
  ret i32 1

}

!0 = !{!"branch_weights", i32 42, i32 99}
