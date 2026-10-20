; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; Threadgroup variables are laid out in the order of the module, whatever
; order the kernel uses them in (PTX programs compute shared memory
; addresses from that layout), and the ones without a size (dynamic shared
; memory) come after all the others.
@buf = internal addrspace(3) global [256 x i8] undef, align 64
@dyn = external addrspace(3) global [0 x i8], align 16
@slot = internal addrspace(3) global [2 x i32] undef, align 16

; CHECK-LABEL: layout:
; CHECK:       // threadgroup memory: 272 bytes
; CHECK:       mov_imm [[R:r[0-9]+]], 256
; CHECK-NEXT:  device_store 0, i32, x, [[R]], u0_u1, 0, signed
; CHECK:       mov_imm [[R:r[0-9]+]], 272
; CHECK-NEXT:  device_store 0, i32, x, [[R]], u0_u1, 1, signed
; CHECK:       mov_imm [[R:r[0-9]+]], 0
; CHECK-NEXT:  device_store 0, i32, x, [[R]], u0_u1, 2, signed
define agx_kernel void @layout(ptr addrspace(1) %o) {
  %s = ptrtoint ptr addrspace(3) @slot to i32
  %d = ptrtoint ptr addrspace(3) @dyn to i32
  %b = ptrtoint ptr addrspace(3) @buf to i32
  store i32 %s, ptr addrspace(1) %o
  %o1 = getelementptr i32, ptr addrspace(1) %o, i32 1
  store i32 %d, ptr addrspace(1) %o1
  %o2 = getelementptr i32, ptr addrspace(1) %o, i32 2
  store i32 %b, ptr addrspace(1) %o2
  ret void
}
