; RUN: opt < %s -passes='dse,verify<memoryssa>' -S | FileCheck %s

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind

; This should create a declaration for the named variant
define ptr @undeclared_customalloc(i64 %size, i64 %align) {
; CHECK-LABEL: @undeclared_customalloc
; CHECK-NEXT:  [[CALL:%.*]] = call ptr @customalloc2_zeroed(i64 [[SIZE:%.*]], i64 [[ALIGN:%.*]])
; CHECK-NEXT:  ret ptr [[CALL]]
  %call = call ptr @customalloc2(i64 %size, i64 %align)
  call void @llvm.memset.p0.i64(ptr %call, i8 0, i64 %size, i1 false)
  ret ptr %call
}

declare ptr @customalloc2(i64, i64) allockind("alloc") "alloc-family"="customalloc2" "alloc-variant-zeroed"="customalloc2_zeroed"
; CHECK-DAG: declare ptr @customalloc2_zeroed(i64, i64) #[[CA2ATTR:[0-9]+]]
; CHECK-DAG: attributes #[[CA2ATTR]] = { allockind("alloc,zeroed") "alloc-family"="customalloc2" }
