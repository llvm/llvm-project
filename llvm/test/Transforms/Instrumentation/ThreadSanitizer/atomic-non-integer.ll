; RUN: opt < %s -passes=tsan -S | FileCheck %s
; Check that atomic memory operations on floating-point types are converted to calls into ThreadSanitizer runtime.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define float @load_float(ptr %fptr) {
  %v = load atomic float, ptr %fptr unordered, align 4
  ret float %v
  ; CHECK-LABEL: load_float
  ; CHECK: call i32 @__tsan_atomic32_load(ptr %{{.+}}, i32 0)
  ; CHECK: bitcast i32 {{.+}} to float
}

define double @load_double(ptr %fptr) {
  %v = load atomic double, ptr %fptr unordered, align 8
  ret double %v
  ; CHECK-LABEL: load_double
  ; CHECK: call i64 @__tsan_atomic64_load(ptr %{{.+}}, i32 0)
  ; CHECK: bitcast i64 {{.+}} to double
}

define fp128 @load_fp128(ptr %fptr) {
  %v = load atomic fp128, ptr %fptr unordered, align 16
  ret fp128 %v
  ; CHECK-LABEL: load_fp128
  ; CHECK: call i128 @__tsan_atomic128_load(ptr %{{.+}}, i32 0)
  ; CHECK: bitcast i128 {{.+}} to fp128
}

define void @store_float(ptr %fptr, float %v) {
  store atomic float %v, ptr %fptr unordered, align 4
  ret void
  ; CHECK-LABEL: store_float
  ; CHECK: bitcast float %v to i32
  ; CHECK: call void @__tsan_atomic32_store(ptr %{{.+}}, i32 %{{.+}}, i32 0)
}

define void @store_double(ptr %fptr, double %v) {
  store atomic double %v, ptr %fptr unordered, align 8
  ret void
  ; CHECK-LABEL: store_double
  ; CHECK: bitcast double %v to i64
  ; CHECK: call void @__tsan_atomic64_store(ptr %{{.+}}, i64 %{{.+}}, i32 0)
}

define void @store_fp128(ptr %fptr, fp128 %v) {
  store atomic fp128 %v, ptr %fptr unordered, align 16
  ret void
  ; CHECK-LABEL: store_fp128
  ; CHECK: bitcast fp128 %v to i128
  ; CHECK: call void @__tsan_atomic128_store(ptr %{{.+}}, i128 %{{.+}}, i32 0)
}
