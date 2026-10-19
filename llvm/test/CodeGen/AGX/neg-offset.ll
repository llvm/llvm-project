; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s

; A negative constant offset is a signed immediate index.
; CHECK-LABEL: neg_offset:
; CHECK: device_store {{.*}}, u0_u1, -1, signed
define agx_kernel void @neg_offset(ptr addrspace(1) %p) {
  %q = getelementptr inbounds i8, ptr addrspace(1) %p, i64 -4
  store i32 22, ptr addrspace(1) %q, align 4
  ret void
}
