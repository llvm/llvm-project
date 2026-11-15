; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64-p1:8:8-p2:16:16-p3:64:64:64:32"

; The element size must not be truncated to 32 bits for a 64-bit index.
; The pointers coincide for x == 0, but differ for x == 1.
define void @large_element(ptr %p, i64 %x) {
; CHECK-LABEL: Function: large_element:
; CHECK: MayAlias: i8* %a, i8* %b
  %a = getelementptr [4294967300 x i8], ptr %p, i64 %x
  %b = getelementptr i32, ptr %p, i64 %x
  load i8, ptr %a
  load i8, ptr %b
  ret void
}

; In particular, a nonzero element size must not become zero.
define void @large_element_zero_low_bits(ptr %p, i64 %x) {
; CHECK-LABEL: Function: large_element_zero_low_bits:
; CHECK: MayAlias: i8* %a, i8* %b
  %a = getelementptr [4294967296 x i8], ptr %p, i64 %x
  %b = getelementptr i8, ptr %p, i64 0
  load i8, ptr %a
  load i8, ptr %b
  ret void
}

; Constant indices already use the full element size.
define void @constant_index(ptr %p) {
; CHECK-LABEL: Function: constant_index:
; CHECK: NoAlias: i8* %a, i8* %b
  %a = getelementptr [4294967300 x i8], ptr %p, i64 1
  %b = getelementptr i8, ptr %p, i64 4
  load i8, ptr %a
  load i8, ptr %b
  ret void
}

; Element sizes wider than the index type must be truncated without asserting.
define void @index8(ptr addrspace(1) %p, i8 %x) {
; CHECK-LABEL: Function: index8:
; CHECK: NoAlias: i8 addrspace(1)* %a, i8 addrspace(1)* %b
  %a = getelementptr [300 x i8], ptr addrspace(1) %p, i8 %x
  %b = getelementptr i8, ptr addrspace(1) %p, i8 1
  load i8, ptr addrspace(1) %a
  load i8, ptr addrspace(1) %b
  ret void
}

; 300 * x and 44 * x are equal modulo 2^8.
define void @index8_equivalent(ptr addrspace(1) %p, i8 %x) {
; CHECK-LABEL: Function: index8_equivalent:
; CHECK: MustAlias: i8 addrspace(1)* %a, i8 addrspace(1)* %b
  %a = getelementptr [300 x i8], ptr addrspace(1) %p, i8 %x
  %offset = mul i8 %x, 44
  %b = getelementptr i8, ptr addrspace(1) %p, i8 %offset
  load i8, ptr addrspace(1) %a
  load i8, ptr addrspace(1) %b
  ret void
}

define void @index16(ptr addrspace(2) %p, i16 %x) {
; CHECK-LABEL: Function: index16:
; CHECK: NoAlias: i8 addrspace(2)* %a, i8 addrspace(2)* %b
  %a = getelementptr [70000 x i8], ptr addrspace(2) %p, i16 %x
  %b = getelementptr i8, ptr addrspace(2) %p, i16 1
  load i8, ptr addrspace(2) %a
  load i8, ptr addrspace(2) %b
  ret void
}

; Use the index width, even when the pointer representation is wider.
define void @index32_pointer64(ptr addrspace(3) %p, i32 %x) {
; CHECK-LABEL: Function: index32_pointer64:
; CHECK: MustAlias: i8 addrspace(3)* %a, i8 addrspace(3)* %b
  %a = getelementptr [4294967300 x i8], ptr addrspace(3) %p, i32 %x
  %b = getelementptr i32, ptr addrspace(3) %p, i32 %x
  load i8, ptr addrspace(3) %a
  load i8, ptr addrspace(3) %b
  ret void
}
