; RUN: opt < %s -passes=lowfat -S | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"

; Test 1: Load should be instrumented with __lf_check_bounds
define i32 @test_load(ptr %p) {
; CHECK-LABEL: @test_load
; CHECK: %[[PTR:.*]] = ptrtoint ptr %p to i64
; CHECK-NEXT: call void @__lf_check_bounds(i64 %[[PTR]], i64 4)
; CHECK-NEXT: %val = load i32, ptr %p
  %val = load i32, ptr %p, align 4
  ret i32 %val
}

; Test 2: Store should be instrumented
define void @test_store(ptr %p, i32 %v) {
; CHECK-LABEL: @test_store
; CHECK: %[[PTR:.*]] = ptrtoint ptr %p to i64
; CHECK-NEXT: call void @__lf_check_bounds(i64 %[[PTR]], i64 4)
; CHECK-NEXT: store i32 %v, ptr %p
  store i32 %v, ptr %p, align 4
  ret void
}

; Test 3: Volatile accesses should NOT be instrumented
define i32 @test_volatile_load(ptr %p) {
; CHECK-LABEL: @test_volatile_load
; CHECK-NOT: call void @__lf_check_bounds
; CHECK: load volatile i32, ptr %p
  %val = load volatile i32, ptr %p, align 4
  ret i32 %val
}

define void @test_volatile_store(ptr %p, i32 %v) {
; CHECK-LABEL: @test_volatile_store
; CHECK-NOT: call void @__lf_check_bounds
; CHECK: store volatile i32 %v, ptr %p
  store volatile i32 %v, ptr %p, align 4
  ret void
}

; Test 4: Runtime functions (__lf_*) should NOT be instrumented
define void @__lf_check_bounds(i64 %ptr, i64 %size) {
; CHECK-LABEL: @__lf_check_bounds
; CHECK-NOT: call void @__lf_check_bounds
  ret void
}

; Test 5: i8 load should use size 1
define i8 @test_load_i8(ptr %p) {
; CHECK-LABEL: @test_load_i8
; CHECK: call void @__lf_check_bounds(i64 %{{.*}}, i64 1)
  %val = load i8, ptr %p, align 1
  ret i8 %val
}

; Test 6: i64 store should use size 8
define void @test_store_i64(ptr %p, i64 %v) {
; CHECK-LABEL: @test_store_i64
; CHECK: call void @__lf_check_bounds(i64 %{{.*}}, i64 8)
  store i64 %v, ptr %p, align 8
  ret void
}

; Test 7: AtomicRMW should be instrumented
define i32 @test_atomic_rmw(ptr %p) {
; CHECK-LABEL: @test_atomic_rmw
; CHECK: call void @__lf_check_bounds(i64 %{{.*}}, i64 4)
; CHECK: atomicrmw add ptr %p, i32 1
  %old = atomicrmw add ptr %p, i32 1 monotonic
  ret i32 %old
}

; Test 8: AtomicCmpXchg should be instrumented
define { i32, i1 } @test_atomic_cmpxchg(ptr %p) {
; CHECK-LABEL: @test_atomic_cmpxchg
; CHECK: call void @__lf_check_bounds(i64 %{{.*}}, i64 4)
; CHECK: cmpxchg ptr %p, i32 0, i32 1
  %val = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic
  ret { i32, i1 } %val
}

; Test 9: Multiple accesses in one function
define void @test_multiple(ptr %p, ptr %q) {
; CHECK-LABEL: @test_multiple
; CHECK: call void @__lf_check_bounds(i64 %{{.*}}, i64 4)
; CHECK: load i32
; CHECK: call void @__lf_check_bounds(i64 %{{.*}}, i64 4)
; CHECK: store i32
  %val = load i32, ptr %p, align 4
  store i32 %val, ptr %q, align 4
  ret void
}
