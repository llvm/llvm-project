; RUN: llc -O2 -mtriple=amdgcn -mcpu=gfx1200 < %s | FileCheck %s
; RUN: llc -O2 -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs < %s -o /dev/null

; "VGPR as memory" objects (allocas in addrspace(13)) accessed at constant
; indices must lower to register copies, never to scratch/buffer memory traffic.

; CHECK-LABEL: store_load_i32:
; CHECK-NOT: scratch_
; CHECK-NOT: buffer_
; CHECK: s_setpc_b64
define i32 @store_load_i32(i32 %v) {
  %a = alloca i32, align 4, addrspace(13)
  store i32 %v, ptr addrspace(13) %a
  %l = load i32, ptr addrspace(13) %a
  %r = add i32 %l, 1
  ret i32 %r
}

; CHECK-LABEL: store_load_array:
; CHECK-NOT: scratch_
; CHECK-NOT: buffer_
; CHECK: s_setpc_b64
define i32 @store_load_array(i32 %v) {
  %a = alloca [4 x i32], align 4, addrspace(13)
  %p1 = getelementptr i32, ptr addrspace(13) %a, i32 1
  %p3 = getelementptr i32, ptr addrspace(13) %a, i32 3
  store i32 %v, ptr addrspace(13) %p1
  store i32 7, ptr addrspace(13) %p3
  %l1 = load i32, ptr addrspace(13) %p1
  %l3 = load i32, ptr addrspace(13) %p3
  %s = add i32 %l1, %l3
  ret i32 %s
}

; A 64-bit (two-dword) access is split into per-dword register copies.
; CHECK-LABEL: store_load_i64:
; CHECK-NOT: scratch_
; CHECK-NOT: buffer_
; CHECK: s_setpc_b64
define i64 @store_load_i64(i64 %v) {
  %a = alloca i64, align 8, addrspace(13)
  store i64 %v, ptr addrspace(13) %a
  %l = load i64, ptr addrspace(13) %a
  %r = add i64 %l, 1
  ret i64 %r
}

; A vector (four-dword) access is split into per-dword register copies.
; CHECK-LABEL: store_load_v4i32:
; CHECK-NOT: scratch_
; CHECK-NOT: buffer_
; CHECK: s_setpc_b64
define <4 x i32> @store_load_v4i32(<4 x i32> %v) {
  %a = alloca <4 x i32>, align 16, addrspace(13)
  store <4 x i32> %v, ptr addrspace(13) %a
  %l = load <4 x i32>, ptr addrspace(13) %a
  ret <4 x i32> %l
}
