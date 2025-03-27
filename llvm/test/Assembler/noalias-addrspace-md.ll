; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define i64 @atomicrmw_noalias_addrspace__0_1(ptr %ptr, i64 %val) {
; CHECK-LABEL: define i64 @atomicrmw_noalias_addrspace__0_1(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]]) {
; CHECK-NEXT:    [[RET:%.*]] = atomicrmw add ptr [[PTR]], i64 [[VAL]] seq_cst, align 8, !noalias.addrspace [[META0:![0-9]+]]
; CHECK-NEXT:    ret i64 [[RET]]
;
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, align 8, !noalias.addrspace !0
  ret i64 %ret
}

define i64 @atomicrmw_noalias_addrspace__0_2(ptr %ptr, i64 %val) {
; CHECK-LABEL: define i64 @atomicrmw_noalias_addrspace__0_2(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]]) {
; CHECK-NEXT:    [[RET:%.*]] = atomicrmw add ptr [[PTR]], i64 [[VAL]] seq_cst, align 8, !noalias.addrspace [[META1:![0-9]+]]
; CHECK-NEXT:    ret i64 [[RET]]
;
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, align 8, !noalias.addrspace !1
  ret i64 %ret
}

define i64 @atomicrmw_noalias_addrspace__1_3(ptr %ptr, i64 %val) {
; CHECK-LABEL: define i64 @atomicrmw_noalias_addrspace__1_3(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]]) {
; CHECK-NEXT:    [[RET:%.*]] = atomicrmw add ptr [[PTR]], i64 [[VAL]] seq_cst, align 8, !noalias.addrspace [[META2:![0-9]+]]
; CHECK-NEXT:    ret i64 [[RET]]
;
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, align 8, !noalias.addrspace !2
  ret i64 %ret
}

define i64 @atomicrmw_noalias_addrspace__multiple_ranges(ptr %ptr, i64 %val) {
; CHECK-LABEL: define i64 @atomicrmw_noalias_addrspace__multiple_ranges(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]]) {
; CHECK-NEXT:    [[RET:%.*]] = atomicrmw add ptr [[PTR]], i64 [[VAL]] seq_cst, align 8, !noalias.addrspace [[META3:![0-9]+]]
; CHECK-NEXT:    ret i64 [[RET]]
;
  %ret = atomicrmw add ptr %ptr, i64 %val seq_cst, align 8, !noalias.addrspace !3
  ret i64 %ret
}

define i64 @load_noalias_addrspace__5_6(ptr %ptr) {
; CHECK-LABEL: define i64 @load_noalias_addrspace__5_6(
; CHECK-SAME: ptr [[PTR:%.*]]) {
; CHECK-NEXT:    [[RET:%.*]] = load i64, ptr [[PTR]], align 4, !noalias.addrspace [[META4:![0-9]+]]
; CHECK-NEXT:    ret i64 [[RET]]
;
  %ret = load i64, ptr %ptr, align 4, !noalias.addrspace !4
  ret i64 %ret
}

define void @store_noalias_addrspace__5_6(ptr %ptr, i64 %val) {
; CHECK-LABEL: define void @store_noalias_addrspace__5_6(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL:%.*]]) {
; CHECK-NEXT:    store i64 [[VAL]], ptr [[PTR]], align 4, !noalias.addrspace [[META4]]
; CHECK-NEXT:    ret void
;
  store i64 %val, ptr %ptr, align 4, !noalias.addrspace !4
  ret void
}

define { i64, i1 } @cmpxchg_noalias_addrspace__5_6(ptr %ptr, i64 %val0, i64 %val1) {
; CHECK-LABEL: define { i64, i1 } @cmpxchg_noalias_addrspace__5_6(
; CHECK-SAME: ptr [[PTR:%.*]], i64 [[VAL0:%.*]], i64 [[VAL1:%.*]]) {
; CHECK-NEXT:    [[RET:%.*]] = cmpxchg ptr [[PTR]], i64 [[VAL0]], i64 [[VAL1]] monotonic monotonic, align 8, !noalias.addrspace [[META4]]
; CHECK-NEXT:    ret { i64, i1 } [[RET]]
;
  %ret = cmpxchg ptr %ptr, i64 %val0, i64 %val1 monotonic monotonic, align 8, !noalias.addrspace !4
  ret { i64, i1 } %ret
}

declare void @foo()

define void @call_noalias_addrspace__5_6(ptr %ptr) {
; CHECK-LABEL: define void @call_noalias_addrspace__5_6(
; CHECK-SAME: ptr [[PTR:%.*]]) {
; CHECK-NEXT:    call void @foo(), !noalias.addrspace [[META4]]
; CHECK-NEXT:    ret void
;
  call void @foo(), !noalias.addrspace !4
  ret void
}

define void @call_memcpy_intrinsic_addrspace__5_6(ptr %dst, ptr %src, i64 %size) {
; CHECK-LABEL: define void @call_memcpy_intrinsic_addrspace__5_6(
; CHECK-SAME: ptr [[DST:%.*]], ptr [[SRC:%.*]], i64 [[SIZE:%.*]]) {
; CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr [[DST]], ptr [[SRC]], i64 [[SIZE]], i1 false), !noalias.addrspace [[META4]]
; CHECK-NEXT:    ret void
;
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %size, i1 false), !noalias.addrspace !4
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!0 = !{i32 0, i32 1}
!1 = !{i32 0, i32 2}
!2 = !{i32 1, i32 3}
!3 = !{i32 4, i32 6, i32 10, i32 55}
!4 = !{i32 5, i32 6}
;.
; CHECK: [[META0]] = !{i32 0, i32 1}
; CHECK: [[META1]] = !{i32 0, i32 2}
; CHECK: [[META2]] = !{i32 1, i32 3}
; CHECK: [[META3]] = !{i32 4, i32 6, i32 10, i32 55}
; CHECK: [[META4]] = !{i32 5, i32 6}
;.
