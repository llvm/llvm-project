; RUN: opt < %s -passes=tsan -S | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

declare void @objc_setProperty_nonatomic(ptr, ptr, ptr, i64) nounwind
declare void @objc_setProperty_nonatomic_copy(ptr, ptr, ptr, i64) nounwind
declare void @objc_setProperty_atomic_copy(ptr, ptr, ptr, i64) nounwind
declare void @objc_setProperty(ptr, ptr, i64, ptr, i1, i1) nounwind

; CHECK-LABEL: define void @test_set_property_nonatomic_copy(
; CHECK: [[ADDR:%.*]] = getelementptr i8, ptr %self, i64 %offset
; CHECK-NEXT: call void @__tsan_write8(ptr [[ADDR]])
; CHECK-NEXT: call void @objc_setProperty_nonatomic_copy(ptr %self, ptr %cmd, ptr %val, i64 %offset)
define void @test_set_property_nonatomic_copy(ptr %self, ptr %cmd, ptr %val, i64 %offset) nounwind sanitize_thread {
  call void @objc_setProperty_nonatomic_copy(ptr %self, ptr %cmd, ptr %val, i64 %offset)
  ret void
}

; CHECK-LABEL: define void @test_set_property_nonatomic(
; CHECK: [[ADDR:%.*]] = getelementptr i8, ptr %self, i64 %offset
; CHECK-NEXT: call void @__tsan_write8(ptr [[ADDR]])
; CHECK-NEXT: call void @objc_setProperty_nonatomic(ptr %self, ptr %cmd, ptr %val, i64 %offset)
define void @test_set_property_nonatomic(ptr %self, ptr %cmd, ptr %val, i64 %offset) nounwind sanitize_thread {
  call void @objc_setProperty_nonatomic(ptr %self, ptr %cmd, ptr %val, i64 %offset)
  ret void
}

; CHECK-LABEL: define void @test_set_property_generic_nonatomic(
; CHECK: [[ADDR:%.*]] = getelementptr i8, ptr %self, i64 %offset
; CHECK-NEXT: call void @__tsan_write8(ptr [[ADDR]])
; CHECK-NEXT: call void @objc_setProperty(ptr %self, ptr %cmd, i64 %offset, ptr %val, i1 false, i1 true)
define void @test_set_property_generic_nonatomic(ptr %self, ptr %cmd, i64 %offset, ptr %val) nounwind sanitize_thread {
  call void @objc_setProperty(ptr %self, ptr %cmd, i64 %offset, ptr %val, i1 false, i1 true)
  ret void
}

; CHECK-LABEL: define void @test_set_property_generic_atomic(
; CHECK-NOT: __tsan_write8
; CHECK: call void @objc_setProperty(ptr %self, ptr %cmd, i64 %offset, ptr %val, i1 true, i1 true)
define void @test_set_property_generic_atomic(ptr %self, ptr %cmd, i64 %offset, ptr %val) nounwind sanitize_thread {
  call void @objc_setProperty(ptr %self, ptr %cmd, i64 %offset, ptr %val, i1 true, i1 true)
  ret void
}

; Clang never emits a non-constant 'atomic' argument for synthesized setters,
; but user code could theoretically call objc_setProperty directly with a
; runtime boolean.
; CHECK-LABEL: define void @test_set_property_generic_nonconstant_atomic(
; CHECK-NOT: __tsan_write8
; CHECK: call void @objc_setProperty(ptr %self, ptr %cmd, i64 %offset, ptr %val, i1 %atomic, i1 true)
define void @test_set_property_generic_nonconstant_atomic(ptr %self, ptr %cmd, i64 %offset, ptr %val, i1 %atomic) nounwind sanitize_thread {
  call void @objc_setProperty(ptr %self, ptr %cmd, i64 %offset, ptr %val, i1 %atomic, i1 true)
  ret void
}

; CHECK-LABEL: define void @test_set_property_atomic_copy(
; CHECK-NOT: __tsan_write8
; CHECK: call void @objc_setProperty_atomic_copy(ptr %self, ptr %cmd, ptr %val, i64 %offset)
define void @test_set_property_atomic_copy(ptr %self, ptr %cmd, ptr %val, i64 %offset) nounwind sanitize_thread {
  call void @objc_setProperty_atomic_copy(ptr %self, ptr %cmd, ptr %val, i64 %offset)
  ret void
}

; CHECK-LABEL: define void @test_no_sanitize_thread(
; CHECK-NOT: __tsan_write8
; CHECK: call void @objc_setProperty_nonatomic_copy(ptr %self, ptr %cmd, ptr %val, i64 %offset)
define void @test_no_sanitize_thread(ptr %self, ptr %cmd, ptr %val, i64 %offset) nounwind {
  call void @objc_setProperty_nonatomic_copy(ptr %self, ptr %cmd, ptr %val, i64 %offset)
  ret void
}
