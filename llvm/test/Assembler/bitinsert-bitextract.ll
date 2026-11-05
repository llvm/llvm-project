; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "p:64:64:64:64-p1:128:128:128:64"

; CHECK-LABEL: test_bitextract
; CHECK: bitextract i8, b32 %src, i32 24
define i8 @test_bitextract(b32 %src) {
  %result = bitextract i8, b32 %src, i32 24
  ret i8 %result
}

; CHECK-LABEL: test_bitinsert
; CHECK: bitinsert b32 %base, i8 %val, i32 3
define b32 @test_bitinsert(b32 %base, i8 %val) {
  %result = bitinsert b32 %base, i8 %val, i32 3
  ret b32 %result
}

; CHECK-LABEL: bitinsert_val_int
; CHECK: %r = bitinsert b64 %base, i32 %val, i32 0
define b64 @bitinsert_val_int(b64 %base, i32 %val) {
  %r = bitinsert b64 %base, i32 %val, i32 0
  ret b64 %r
}

; CHECK-LABEL: bitinsert_val_fp
; CHECK: %r = bitinsert b64 %base, double %val, i32 0
define b64 @bitinsert_val_fp(b64 %base, double %val) {
  %r = bitinsert b64 %base, double %val, i32 0
  ret b64 %r
}

; CHECK-LABEL: bitinsert_val_byte
; CHECK: %r = bitinsert b64 %base, b32 %val, i32 0
define b64 @bitinsert_val_byte(b64 %base, b32 %val) {
  %r = bitinsert b64 %base, b32 %val, i32 0
  ret b64 %r
}

; CHECK-LABEL: bitinsert_val_ptr
; CHECK: %r = bitinsert b64 %base, ptr %val, i32 0
define b64 @bitinsert_val_ptr(b64 %base, ptr %val) {
  %r = bitinsert b64 %base, ptr %val, i32 0
  ret b64 %r
}

; CHECK-LABEL: bitinsert_val_ptr_other_as
; CHECK: %r = bitinsert b64 %base, ptr addrspace(2) %val, i32 0
define b64 @bitinsert_val_ptr_other_as(b64 %base, ptr addrspace(2) %val) {
  %r = bitinsert b64 %base, ptr addrspace(2) %val, i32 0
  ret b64 %r
}

; CHECK-LABEL: bitinsert_val_non_address_bits
; CHECK: %r = bitinsert b128 %base, ptr addrspace(1) %val, i32 0
define b128 @bitinsert_val_non_address_bits(b128 %base,
                                              ptr addrspace(1) %val) {
  %r = bitinsert b128 %base, ptr addrspace(1) %val, i32 0
  ret b128 %r
}

; CHECK-LABEL: bitextract_ty_int
; CHECK: %r = bitextract i32, b64 %src, i32 0
define i32 @bitextract_ty_int(b64 %src) {
  %r = bitextract i32, b64 %src, i32 0
  ret i32 %r
}

; CHECK-LABEL: bitextract_ty_fp
; CHECK: %r = bitextract double, b64 %src, i32 0
define double @bitextract_ty_fp(b64 %src) {
  %r = bitextract double, b64 %src, i32 0
  ret double %r
}

; CHECK-LABEL: bitextract_ty_byte
; CHECK: %r = bitextract b32, b64 %src, i32 0
define b32 @bitextract_ty_byte(b64 %src) {
  %r = bitextract b32, b64 %src, i32 0
  ret b32 %r
}

; CHECK-LABEL: bitextract_ty_ptr
; CHECK: %r = bitextract ptr, b64 %src, i32 0
define ptr @bitextract_ty_ptr(b64 %src) {
  %r = bitextract ptr, b64 %src, i32 0
  ret ptr %r
}

; CHECK-LABEL: bitextract_ty_ptr_other_as
; CHECK: %r = bitextract ptr addrspace(2), b64 %src, i32 0
define ptr addrspace(2) @bitextract_ty_ptr_other_as(b64 %src) {
  %r = bitextract ptr addrspace(2), b64 %src, i32 0
  ret ptr addrspace(2) %r
}

; CHECK-LABEL: bitextract_ty_non_address_bits
; CHECK: %r = bitextract ptr addrspace(1), b128 %src, i32 0
define ptr addrspace(1) @bitextract_ty_non_address_bits(b128 %src) {
  %r = bitextract ptr addrspace(1), b128 %src, i32 0
  ret ptr addrspace(1) %r
}
