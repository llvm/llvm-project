; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; Verify the import works if the blocks are not topologically sorted.
; CHECK-LABEL: @dominance_order
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define i64 @dominance_order(i64 %arg1) {
  ; CHECK: llvm.br ^[[BB2:.+]]
  br label %bb2
bb1:
  ; CHECK: ^[[BB1:[a-zA-Z0-9]+]]:
  ; CHECK:  llvm.return %[[VAL1:.+]] : i64
  ret i64 %1
bb2:
  ; CHECK: ^[[BB2]]:
  ; CHECK: %[[VAL1]] = llvm.add %[[ARG1]]
  %1 = add i64 %arg1, 3
  ; CHECK: llvm.br ^[[BB1]]
  br label %bb1
}

; // -----

; CHECK-LABEL: @block_argument
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
define i64 @block_argument(i1 %arg1, i64 %arg2) {
entry:
  ; CHECK: llvm.cond_br %[[ARG1]]
  ; CHECK-SAME: ^[[BB1:.+]](%[[ARG2]] : i64)
  ; CHECK-SAME: ^[[BB2:.+]]
  br i1 %arg1, label %bb1, label %bb2
bb1:
  ; CHECK: ^[[BB1]](%[[BA1:.+]]: i64):
  ; CHECK: llvm.return %[[BA1]] : i64
  %0 = phi i64 [ %arg2, %entry ], [ %1, %bb2 ]
  ret i64 %0
bb2:
  ; CHECK: ^[[BB2]]:
  ; CHECK: %[[VAL1:.+]] = llvm.add %[[ARG2]]
  ; CHECK: llvm.br ^[[BB1]](%[[VAL1]]
  %1 = add i64 %arg2, 3
  br label %bb1
}

; // -----

; CHECK-LABEL: @simple_switch(
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define i64 @simple_switch(i64 %arg1) {
  ; CHECK: %[[VAL1:.+]] = llvm.add
  ; CHECK: %[[VAL2:.+]] = llvm.sub
  ; CHECK: %[[VAL3:.+]] = llvm.mul
  %1 = add i64 %arg1, 42
  %2 = sub i64 %arg1, 42
  %3 = mul i64 %arg1, 42
  ; CHECK: llvm.switch %[[ARG1]] : i64, ^[[BBD:.+]] [
  ; CHECK:   0: ^[[BB1:.+]],
  ; CHECK:   9: ^[[BB2:.+]]
  ; CHECK: ]
  switch i64 %arg1, label %bbd [
    i64 0, label %bb1
    i64 9, label %bb2
  ]
bb1:
  ; CHECK: ^[[BB1]]:
  ; CHECK: llvm.return %[[VAL1]]
  ret i64 %1
bb2:
  ; CHECK: ^[[BB2]]:
  ; CHECK: llvm.return %[[VAL2]]
  ret i64 %2
bbd:
  ; CHECK: ^[[BBD]]:
  ; CHECK: llvm.return %[[VAL3]]
  ret i64 %3
}

; // -----

; CHECK-LABEL: @switch_args
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define i32 @switch_args(i32 %arg1) {
entry:
  ; CHECK: %[[VAL1:.+]] = llvm.add
  ; CHECK: %[[VAL2:.+]] = llvm.sub
  ; CHECK: %[[VAL3:.+]] = llvm.mul
  %0 = add i32 %arg1, 42
  %1 = sub i32 %arg1, 42
  %2 = mul i32 %arg1, 42
  ; CHECK: llvm.switch %[[ARG1]] : i32, ^[[BBD:.+]](%[[VAL3]] : i32) [
  ; CHECK:   0: ^[[BB1:.+]](%[[VAL1]], %[[VAL2]] : i32, i32)
  ; CHECK: ]
  switch i32 %arg1, label %bbd [
    i32 0, label %bb1
  ]
bb1:
  ; CHECK: ^[[BB1]](%[[BA1:.+]]: i32, %[[BA2:.+]]: i32):
  ; CHECK: %[[VAL1:.*]] = llvm.add %[[BA1]], %[[BA2]] : i32
  %3 = phi i32 [%0, %entry]
  %4 = phi i32 [%1, %entry]
  %5 = add i32 %3, %4
  ; CHECK: llvm.br ^[[BBD]](%[[VAL1]]
  br label %bbd
bbd:
  ; CHECK: ^[[BBD]](%[[BA3:.+]]: i32):
  ; CHECK: llvm.return %[[BA3]]
  %6 = phi i32 [%2, %entry], [%5, %bb1]
  ret i32 %6
}
