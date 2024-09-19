; RUN: opt < %s -S -msan-kernel=1 -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le--linux"

define void @Store1(ptr %p, i8 %x) sanitize_memory {
entry:
  store i8 %x, ptr %p
  ret void
}

; CHECK-LABEL: define {{[^@]+}}@Store1(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_1(ptr %p)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: store i8 {{.+}}, ptr [[SHADOW]]
; CHECK: ret void

define void @Store2(ptr %p, i16 %x) sanitize_memory {
entry:
  store i16 %x, ptr %p
  ret void
}

; CHECK-LABEL: define {{[^@]+}}@Store2(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_2(ptr %p)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: store i16 {{.+}}, ptr [[SHADOW]]
; CHECK: ret void

define void @Store4(ptr %p, i32 %x) sanitize_memory {
entry:
  store i32 %x, ptr %p
  ret void
}

; CHECK-LABEL: define {{[^@]+}}@Store4(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_4(ptr %p)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: store i32 {{.+}}, ptr [[SHADOW]]
; CHECK: ret void

define void @Store8(ptr %p, i64 %x) sanitize_memory {
entry:
  store i64 %x, ptr %p
  ret void
}

; CHECK-LABEL: define {{[^@]+}}@Store8(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: store i64 {{.+}}, ptr [[SHADOW]]
; CHECK: ret void

define void @Store16(ptr %p, i128 %x) sanitize_memory {
entry:
  store i128 %x, ptr %p
  ret void
}

; CHECK-LABEL: define {{[^@]+}}@Store16(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_n(ptr %p, i64 16)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: store i128 {{.+}}, ptr [[SHADOW]]
; CHECK: ret void

define i8 @Load1(ptr %p) sanitize_memory {
entry:
  %0 = load i8, ptr %p
  ret i8 %0
}

; CHECK-LABEL: define {{[^@]+}}@Load1(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_load_1(ptr %p)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: [[SHADOW_VAL:%[a-z0-9_]+]] = load i8, ptr [[SHADOW]]
; CHECK: [[ORIGIN_VAL:%[a-z0-9_]+]] = load i32, ptr [[ORIGIN]]
; CHECK: store i8 [[SHADOW_VAL]], ptr %retval_shadow
; CHECK: store i32 [[ORIGIN_VAL]], ptr %retval_origin
; CHECK: ret i8 {{.+}}

define i16 @Load2(ptr %p) sanitize_memory {
entry:
  %0 = load i16, ptr %p
  ret i16 %0
}

; CHECK-LABEL: define {{[^@]+}}@Load2(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_load_2(ptr %p)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: [[SHADOW_VAL:%[a-z0-9_]+]] = load i16, ptr [[SHADOW]]
; CHECK: [[ORIGIN_VAL:%[a-z0-9_]+]] = load i32, ptr [[ORIGIN]]
; CHECK: store i16 [[SHADOW_VAL]], ptr %retval_shadow
; CHECK: store i32 [[ORIGIN_VAL]], ptr %retval_origin
; CHECK: ret i16 {{.+}}

define i32 @Load4(ptr %p) sanitize_memory {
entry:
  %0 = load i32, ptr %p
  ret i32 %0
}

; CHECK-LABEL: define {{[^@]+}}@Load4(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_load_4(ptr %p)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: [[SHADOW_VAL:%[a-z0-9_]+]] = load i32, ptr [[SHADOW]]
; CHECK: [[ORIGIN_VAL:%[a-z0-9_]+]] = load i32, ptr [[ORIGIN]]
; CHECK: store i32 [[SHADOW_VAL]], ptr %retval_shadow
; CHECK: store i32 [[ORIGIN_VAL]], ptr %retval_origin
; CHECK: ret i32 {{.+}}

define i64 @Load8(ptr %p) sanitize_memory {
entry:
  %0 = load i64, ptr %p
  ret i64 %0
}

; CHECK-LABEL: define {{[^@]+}}@Load8(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: [[SHADOW_VAL:%[a-z0-9_]+]] = load i64, ptr [[SHADOW]]
; CHECK: [[ORIGIN_VAL:%[a-z0-9_]+]] = load i32, ptr [[ORIGIN]]
; CHECK: store i64 [[SHADOW_VAL]], ptr %retval_shadow
; CHECK: store i32 [[ORIGIN_VAL]], ptr %retval_origin
; CHECK: ret i64 {{.+}}

define i128 @Load16(ptr %p) sanitize_memory {
entry:
  %0 = load i128, ptr %p
  ret i128 %0
}

; CHECK-LABEL: define {{[^@]+}}@Load16(
; CHECK: [[META:%[a-z0-9_]+]] = call { ptr, ptr } @__msan_metadata_ptr_for_load_n(ptr %p, i64 16)
; CHECK: [[SHADOW:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 0
; CHECK: [[ORIGIN:%[a-z0-9_]+]] = extractvalue { ptr, ptr } [[META]], 1
; CHECK: [[SHADOW_VAL:%[a-z0-9_]+]] = load i128, ptr [[SHADOW]]
; CHECK: [[ORIGIN_VAL:%[a-z0-9_]+]] = load i32, ptr [[ORIGIN]]
; CHECK: store i128 [[SHADOW_VAL]], ptr %retval_shadow
; CHECK: store i32 [[ORIGIN_VAL]], ptr %retval_origin
; CHECK: ret i128 {{.+}}
