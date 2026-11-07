; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 | FileCheck %s
; RUN: %if ptxas-sm_80 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 | %ptxas-verify -arch=sm_80 %}

; A memcpy whose source and destination are both 16 byte aligned is expanded
; into 16 byte vector loads and stores, with the loads issued before the
; stores. Lower alignment keeps 8 byte accesses.

; CHECK-LABEL: .entry copy32(
; CHECK: ld.global.v4.b32
; CHECK-NEXT: ld.global.v4.b32
; CHECK-NEXT: st.global.v4.b32
; CHECK-NEXT: st.global.v4.b32
; CHECK-NEXT: ret;
define ptx_kernel void @copy32(ptr addrspace(1) align 32 %dst, ptr addrspace(1) align 32 %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 32 %dst, ptr addrspace(1) align 32 %src, i64 32, i1 false)
  ret void
}

; CHECK-LABEL: .entry copy16(
; CHECK: ld.global.v4.b32
; CHECK-NEXT: st.global.v4.b32
; CHECK-NEXT: ret;
define ptx_kernel void @copy16(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 16 %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 16 %src, i64 16, i1 false)
  ret void
}

; CHECK-LABEL: .entry copy24(
; CHECK-DAG: ld.global.v4.b32
; CHECK-DAG: ld.global.b64
; CHECK-DAG: st.global.v4.b32
; CHECK-DAG: st.global.b64
; CHECK: ret;
define ptx_kernel void @copy24(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 16 %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 16 %src, i64 24, i1 false)
  ret void
}

; CHECK-LABEL: .entry copy32_align8(
; CHECK-NOT: .v4.
; CHECK-COUNT-4: ld.global.b64
; CHECK-NOT: .v4.
; CHECK: ret;
define ptx_kernel void @copy32_align8(ptr addrspace(1) align 8 %dst, ptr addrspace(1) align 8 %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 8 %dst, ptr addrspace(1) align 8 %src, i64 32, i1 false)
  ret void
}

; CHECK-LABEL: .entry copy32_src_align8(
; CHECK-NOT: .v4.
; CHECK: ret;
define ptx_kernel void @copy32_src_align8(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 8 %src) {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 8 %src, i64 32, i1 false)
  ret void
}

; CHECK-LABEL: .func copy32_generic(
; CHECK: ld.v4.b32
; CHECK-NEXT: ld.v4.b32
; CHECK-NEXT: st.v4.b32
; CHECK-NEXT: st.v4.b32
; CHECK-NEXT: ret;
define void @copy32_generic(ptr align 32 %dst, ptr align 32 %src) {
  call void @llvm.memcpy.p0.p0.i64(ptr align 32 %dst, ptr align 32 %src, i64 32, i1 false)
  ret void
}

; CHECK-LABEL: .entry move32(
; CHECK: ld.global.v4.b32
; CHECK-NEXT: ld.global.v4.b32
; CHECK-NEXT: st.global.v4.b32
; CHECK-NEXT: st.global.v4.b32
; CHECK-NEXT: ret;
define ptx_kernel void @move32(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 16 %src) {
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 16 %dst, ptr addrspace(1) align 16 %src, i64 32, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
