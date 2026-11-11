; Tests CopyProf store instrumentation.
;
; RUN: opt < %s -passes='function(copyprof-stores)' -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

;; Tests that a simple i32 store is instrumented.
define void @test_store_i32(ptr %a) {
entry:
  store i32 42, ptr %a, align 4
  ret void
}
; CHECK-LABEL: define void @test_store_i32(ptr %a)
; CHECK:         call void @__copyprof_store_callback(ptr %a, i64 4)
; CHECK-NEXT:    store i32 42, ptr %a

;; Tests that an i64 store is instrumented with the correct size.
define void @test_store_i64(ptr %a) {
entry:
  store i64 100, ptr %a, align 8
  ret void
}
; CHECK-LABEL: define void @test_store_i64(ptr %a)
; CHECK:         call void @__copyprof_store_callback(ptr %a, i64 8)
; CHECK-NEXT:    store i64 100, ptr %a

;; Tests that multiple stores in the same function are all instrumented.
define void @test_multiple_stores(ptr %a, ptr %b) {
entry:
  store i32 1, ptr %a, align 4
  store i32 2, ptr %b, align 4
  ret void
}
; CHECK-LABEL: define void @test_multiple_stores(ptr %a, ptr %b)
; CHECK:         call void @__copyprof_store_callback(ptr %a, i64 4)
; CHECK-NEXT:    store i32 1, ptr %a
; CHECK:         call void @__copyprof_store_callback(ptr %b, i64 4)
; CHECK-NEXT:    store i32 2, ptr %b

;; Tests that a function with no stores is not modified.
define i32 @test_no_stores(ptr %a) {
entry:
  %val = load i32, ptr %a
  ret i32 %val
}
; CHECK-LABEL: define i32 @test_no_stores(ptr %a)
; CHECK-NOT:     call void @__copyprof_store_callback
; CHECK:         ret i32

;; Tests that stores to non-default address spaces are not instrumented.
define void @test_addrspace_store(ptr addrspace(1) %a) {
entry:
  store i32 42, ptr addrspace(1) %a, align 4
  ret void
}
; CHECK-LABEL: define void @test_addrspace_store(ptr addrspace(1) %a)
; CHECK-NOT:     call void @__copyprof_store_callback
; CHECK:         store i32 42, ptr addrspace(1) %a
; CHECK-NEXT:    ret void

;; Tests that scalable vector stores are skipped as they have no
;; compile-time-constant store size.
define void @test_store_scalable_vector(ptr %a) {
entry:
  store <vscale x 4 x i32> zeroinitializer, ptr %a, align 16
  ret void
}
; CHECK-LABEL: define void @test_store_scalable_vector(ptr %a)
; CHECK-NOT:     call void @__copyprof_store_callback
; CHECK:         store <vscale x 4 x i32> zeroinitializer, ptr %a
; CHECK-NEXT:    ret void

;; Tests that a vector store is instrumented with the correct aggregate size.
define void @test_store_vector(ptr %a) {
entry:
  store <4 x i32> zeroinitializer, ptr %a, align 16
  ret void
}
; CHECK-LABEL: define void @test_store_vector(ptr %a)
; CHECK:         call void @__copyprof_store_callback(ptr %a, i64 16)
; CHECK-NEXT:    store <4 x i32> zeroinitializer, ptr %a

;; Tests that an aggregate store is instrumented with the correct size.
define void @test_store_struct(ptr %a) {
entry:
  store <{ i32, i16, i8 }> zeroinitializer, ptr %a
  ret void
}
; CHECK-LABEL: define void @test_store_struct(ptr %a)
; CHECK:         call void @__copyprof_store_callback(ptr %a, i64 7)
; CHECK-NEXT:    store <{ i32, i16, i8 }> zeroinitializer, ptr %a

;; Tests that volatile stores are instrumented.
define void @test_store_volatile(ptr %a) {
entry:
  store volatile i32 42, ptr %a, align 4
  ret void
}
; CHECK-LABEL: define void @test_store_volatile(ptr %a)
; CHECK:         call void @__copyprof_store_callback(ptr %a, i64 4)
; CHECK-NEXT:    store volatile i32 42, ptr %a

;; Tests that atomic stores are instrumented.
define void @test_store_atomic(ptr %a) {
entry:
  store atomic i32 42, ptr %a monotonic, align 4
  ret void
}
; CHECK-LABEL: define void @test_store_atomic(ptr %a)
; CHECK:         call void @__copyprof_store_callback(ptr %a, i64 4)
; CHECK-NEXT:    store atomic i32 42, ptr %a monotonic

