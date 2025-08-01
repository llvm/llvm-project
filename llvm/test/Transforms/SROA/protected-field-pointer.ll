; RUN: opt -passes=sroa -S < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; CHECK: define void @slice
define void @slice(ptr %ptr1, ptr %ptr2, ptr %out1, ptr %out2) {
  %alloca = alloca { ptr, ptr }

  %protptrptr1.1 = call ptr @llvm.protected.field.ptr(ptr %alloca, i64 1, i1 true)
  store ptr %ptr1, ptr %protptrptr1.1
  %protptrptr1.2 = call ptr @llvm.protected.field.ptr(ptr %alloca, i64 1, i1 true)
  %ptr1a = load ptr, ptr %protptrptr1.2

  %gep = getelementptr { ptr, ptr }, ptr %alloca, i64 0, i32 1
  %protptrptr2.1 = call ptr @llvm.protected.field.ptr(ptr %gep, i64 2, i1 true)
  store ptr %ptr2, ptr %protptrptr1.1
  %protptrptr2.2 = call ptr @llvm.protected.field.ptr(ptr %gep, i64 2, i1 true)
  %ptr2a = load ptr, ptr %protptrptr1.2

  ; CHECK-NEXT: store ptr %ptr1, ptr %out1, align 8
  store ptr %ptr1a, ptr %out1
  ; CHECK-NEXT: store ptr %ptr2, ptr %out2, align 8
  store ptr %ptr2a, ptr %out2
  ret void
}

; CHECK: define ptr @mixed
define ptr @mixed(ptr %ptr) {
  ; CHECK-NEXT: %alloca = alloca ptr, align 8
  %alloca = alloca ptr

  ; CHECK-NEXT: store ptr %ptr, ptr %alloca, align 8
  store ptr %ptr, ptr %alloca
  ; CHECK-NEXT: %protptrptr1.2 = call ptr @llvm.protected.field.ptr(ptr %alloca, i64 1, i1 true)
  %protptrptr1.2 = call ptr @llvm.protected.field.ptr(ptr %alloca, i64 1, i1 true)
  ; CHECK-NEXT: %ptr1a = load ptr, ptr %protptrptr1.2, align 8
  %ptr1a = load ptr, ptr %protptrptr1.2

  ; CHECK-NEXT: ret ptr %ptr
  ret ptr %ptr
}
