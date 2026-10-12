; RUN: opt -passes=attributor -verify-each -S %s | FileCheck %s
; RUN: opt -passes=attributor-cgscc -verify-each -S %s | FileCheck %s
;
; The select can refer to offsets 2 and 0 of the same allocation. Collecting
; these offsets can produce a non-sorted SmallSet iteration order. Constant
; vector stores must sort the offsets before constructing a RangeList.
; Both destinations are in bounds; no target-specific behavior is required.

define void @constant_vector_store(i1 %c) {
; CHECK-LABEL: define void @constant_vector_store(
; CHECK-NOT: store
; CHECK: ret void
  %base = alloca [4 x i8]
  %hi = getelementptr i8, ptr %base, i64 2
  %dst = select i1 %c, ptr %hi, ptr %base
  store <2 x i8> <i8 7, i8 9>, ptr %dst, align 1
  ret void
}
