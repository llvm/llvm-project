; RUN: opt < %s -passes=dasan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

; CHECK-LABEL: define void @elements(
; CHECK:         ptrtoint ptr addrspace(1) %p to i64
; CHECK:         load i64, ptr addrspace(1) {{%.*}}, align 8, !alias.scope
; CHECK-NOT:     !alias.scope
; CHECK:         ret void
define void @elements(ptr addrspace(1) %p, i64 %i, i64 %j) sanitize_device_address {
  %a = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %b = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %j
  store i32 0, ptr addrspace(1) %a, align 4
  store i32 1, ptr addrspace(1) %b, align 4
  ret void
}

; CHECK-LABEL: define void @rows(
; CHECK:         %row = getelementptr [4000 x i8], ptr addrspace(1) %p, i64 %r
; CHECK:         [[B:%.*]] = ptrtoint ptr addrspace(1) %row to i64
; CHECK-NOT:     ptrtoint ptr addrspace(1) %p to i64
; CHECK:         ret void
define void @rows(ptr addrspace(1) %p, i64 %r, i64 %i) sanitize_device_address {
  %row = getelementptr [4000 x i8], ptr addrspace(1) %p, i64 %r
  %e = getelementptr inbounds i32, ptr addrspace(1) %row, i64 %i
  store i32 0, ptr addrspace(1) %e, align 4
  ret void
}

; CHECK-LABEL: define void @field(
; CHECK:         ptrtoint ptr addrspace(1) %s to i64
; CHECK:         ret void
define void @field(ptr addrspace(1) %s) sanitize_device_address {
  %f = getelementptr inbounds { i32, [4000 x i8] }, ptr addrspace(1) %s, i64 0, i32 1
  store i8 0, ptr addrspace(1) %f, align 1
  ret void
}

; CHECK-LABEL: define void @wide(
; CHECK:         ptrtoint ptr addrspace(1) %p to i64
; CHECK:         ret void
define void @wide(ptr addrspace(1) %p, i64 %i) sanitize_device_address {
  %a = getelementptr inbounds <4 x float>, ptr addrspace(1) %p, i64 %i
  store <4 x float> zeroinitializer, ptr addrspace(1) %a, align 16
  ret void
}

; CHECK-LABEL: define void @rows2d(
; CHECK:         %e = getelementptr [8 x [4000 x i8]], ptr addrspace(1) %p, i64 0, i64 %r, i64 %i
; CHECK:         ptrtoint ptr addrspace(1) %e to i64
; CHECK-NOT:     ptrtoint ptr addrspace(1) %p to i64
define void @rows2d(ptr addrspace(1) %p, i64 %r, i64 %i) sanitize_device_address {
  %e = getelementptr [8 x [4000 x i8]], ptr addrspace(1) %p, i64 0, i64 %r, i64 %i
  store i8 0, ptr addrspace(1) %e, align 1
  ret void
}

; CHECK-LABEL: define void @rows2d_split(
; CHECK:         %row = getelementptr [8 x [4000 x i8]], ptr addrspace(1) %p, i64 0, i64 %r
; CHECK:         ptrtoint ptr addrspace(1) %row to i64
; CHECK-NOT:     ptrtoint ptr addrspace(1) %p to i64
define void @rows2d_split(ptr addrspace(1) %p, i64 %r, i64 %i) sanitize_device_address {
  %row = getelementptr [8 x [4000 x i8]], ptr addrspace(1) %p, i64 0, i64 %r
  %e = getelementptr inbounds [4000 x i8], ptr addrspace(1) %row, i64 0, i64 %i
  store i32 0, ptr addrspace(1) %e, align 4
  ret void
}
