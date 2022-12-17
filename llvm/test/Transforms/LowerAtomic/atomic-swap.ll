; RUN: opt < %s -passes=loweratomic -S | FileCheck %s

define i8 @cmpswap() {
; CHECK-LABEL: @cmpswap(
  %i = alloca i8
  %pair = cmpxchg ptr %i, i8 0, i8 42 monotonic monotonic
  %j = extractvalue { i8, i1 } %pair, 0
; CHECK: [[OLDVAL:%[a-z0-9]+]] = load i8, ptr [[ADDR:%[a-z0-9]+]]
; CHECK-NEXT: [[SAME:%[a-z0-9]+]] = icmp eq i8 [[OLDVAL]], 0
; CHECK-NEXT: [[TO_STORE:%[a-z0-9]+]] = select i1 [[SAME]], i8 42, i8 [[OLDVAL]]
; CHECK-NEXT: store i8 [[TO_STORE]], ptr [[ADDR]]
; CHECK-NEXT: [[TMP:%[a-z0-9]+]] = insertvalue { i8, i1 } poison, i8 [[OLDVAL]], 0
; CHECK-NEXT: [[RES:%[a-z0-9]+]] = insertvalue { i8, i1 } [[TMP]], i1 [[SAME]], 1
; CHECK-NEXT: [[VAL:%[a-z0-9]+]] = extractvalue { i8, i1 } [[RES]], 0
  ret i8 %j
; CHECK: ret i8 [[VAL]]
}


define i8 @swap() {
; CHECK-LABEL: @swap(
  %i = alloca i8
  %j = atomicrmw xchg ptr %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}


define i8 @swap_optnone() noinline optnone {
; CHECK-LABEL: @swap_optnone(
  %i = alloca i8
  %j = atomicrmw xchg ptr %i, i8 42 monotonic
; CHECK: [[INST:%[a-z0-9]+]] = load
; CHECK-NEXT: store
  ret i8 %j
; CHECK: ret i8 [[INST]]
}
