; RUN: opt -passes=pre-isel-intrinsic-lowering -S < %s | FileCheck --check-prefixes=CHECK,NOPAUTH %s
; RUN: opt -passes=pre-isel-intrinsic-lowering -mattr=+pauth -S < %s | FileCheck --check-prefixes=CHECK,PAUTH %s

target triple = "aarch64-unknown-linux-gnu"

; CHECK: @ds1 = external global i8
@ds1 = external global i8
; CHECK: @ds2 = external global i8
@ds2 = external global i8
; CHECK: @ds3 = hidden alias i8, inttoptr (i64 3573751839 to ptr)
@ds3 = external global i8

; CHECK: define ptr @f1
define ptr @f1(ptr %ptrptr) {
  ; CHECK: %ptr = load ptr, ptr %ptrptr, align 8
  ; CHECK: %1 = ptrtoint ptr %ptr to i64
  ; NOPAUTH: %2 = call i64 @__emupac_autda(i64 %1, i64 1) [ "deactivation-symbol"(ptr @ds1) ]
  ; PAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 1) [ "deactivation-symbol"(ptr @ds1) ]
  ; CHECK: %3 = inttoptr i64 %2 to ptr
  ; CHECK: ret ptr %3
  %protptrptr = call ptr @llvm.protected.field.ptr(ptr %ptrptr, i64 1, i1 true) [ "deactivation-symbol"(ptr @ds1) ]
  %ptr = load ptr, ptr %protptrptr
  ret ptr %ptr
}

; CHECK: define void @f2
define void @f2(ptr %ptrptr, ptr %ptr) {
  ; CHECK: %1 = ptrtoint ptr %ptr to i64
  ; NOPAUTH: %2 = call i64 @__emupac_pacda(i64 %1, i64 2) [ "deactivation-symbol"(ptr @ds2) ]
  ; PAUTH: %2 = call i64 @llvm.ptrauth.sign(i64 %1, i32 2, i64 2) [ "deactivation-symbol"(ptr @ds2) ]
  ; CHECK: %3 = inttoptr i64 %2 to ptr
  ; CHECK: store ptr %3, ptr %ptrptr, align 8
  ; CHECK: ret void
  %protptrptr = call ptr @llvm.protected.field.ptr(ptr %ptrptr, i64 2, i1 true) [ "deactivation-symbol"(ptr @ds2) ]
  store ptr %ptr, ptr %protptrptr
  ret void
}

; CHECK: define ptr @f3
define ptr @f3(ptr %ptrptr) {
  ; CHECK: ret ptr %ptrptr
  %protptrptr = call ptr @llvm.protected.field.ptr(ptr %ptrptr, i64 3, i1 true) [ "deactivation-symbol"(ptr @ds3) ]
  ret ptr %protptrptr
}

declare ptr @llvm.protected.field.ptr(ptr, i64, i1 immarg)
