; RUN: opt < %s -loweratomic -S | FileCheck %s

define i8 @add() {
; CHECK: @add
  %i = alloca i8
  %j = atomicrmw add i8* %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: add
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define i8 @nand() {
; CHECK: @nand
  %i = alloca i8
  %j = atomicrmw nand i8* %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: and
; CHECK-NEXT: xor
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define i8 @min() {
; CHECK: @min
  %i = alloca i8
  %j = atomicrmw min i8* %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}
