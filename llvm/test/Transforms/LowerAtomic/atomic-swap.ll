; RUN: opt < %s -loweratomic -S | FileCheck %s

define i8 @cmpswap() {
; CHECK: @cmpswap
  %i = alloca i8
  %j = cmpxchg i8* %i, i8 0, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: icmp
; CHECK-NEXT: select
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}

define i8 @swap() {
; CHECK: @swap
  %i = alloca i8
  %j = atomicrmw xchg i8* %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}
