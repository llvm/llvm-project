; RUN: opt < %s -tsan -S | FileCheck %s
; Check that atomic memory operations on floating-point types are converted to calls into ThreadSanitizer runtime.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define float @load_float(float* %fptr) {
  %v = load atomic float, float* %fptr unordered, align 4
  ret float %v
  ; CHECK-LABEL: load_float
  ; CHECK: call i32 @__tsan_atomic32_load(i32* %{{.+}}, i32 0)
  ; CHECK: bitcast i32 {{.+}} to float
}

define double @load_double(double* %fptr) {
  %v = load atomic double, double* %fptr unordered, align 8
  ret double %v
  ; CHECK-LABEL: load_double
  ; CHECK: call i64 @__tsan_atomic64_load(i64* %{{.+}}, i32 0)
  ; CHECK: bitcast i64 {{.+}} to double
}

define fp128 @load_fp128(fp128* %fptr) {
  %v = load atomic fp128, fp128* %fptr unordered, align 16
  ret fp128 %v
  ; CHECK-LABEL: load_fp128
  ; CHECK: call i128 @__tsan_atomic128_load(i128* %{{.+}}, i32 0)
  ; CHECK: bitcast i128 {{.+}} to fp128
}

define void @store_float(float* %fptr, float %v) {
  store atomic float %v, float* %fptr unordered, align 4
  ret void
  ; CHECK-LABEL: store_float
  ; CHECK: bitcast float %v to i32
  ; CHECK: call void @__tsan_atomic32_store(i32* %{{.+}}, i32 %{{.+}}, i32 0)
}

define void @store_double(double* %fptr, double %v) {
  store atomic double %v, double* %fptr unordered, align 8
  ret void
  ; CHECK-LABEL: store_double
  ; CHECK: bitcast double %v to i64
  ; CHECK: call void @__tsan_atomic64_store(i64* %{{.+}}, i64 %{{.+}}, i32 0)
}

define void @store_fp128(fp128* %fptr, fp128 %v) {
  store atomic fp128 %v, fp128* %fptr unordered, align 16
  ret void
  ; CHECK-LABEL: store_fp128
  ; CHECK: bitcast fp128 %v to i128
  ; CHECK: call void @__tsan_atomic128_store(i128* %{{.+}}, i128 %{{.+}}, i32 0)
}
