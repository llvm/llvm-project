; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefix=IMM
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -agx-uniform-constants < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-uniform-constants -filetype=obj < %s | llvm-readelf -x .agx.kernels - | FileCheck %s --check-prefix=SECTION

; -agx-uniform-constants: the wide constants of a kernel are preloaded into the
; uniforms after the arguments and threadgroups_per_grid (once per value), and
; listed for the driver in extension 6 of the kernel's record.
; IMM-LABEL: uconst:
; IMM:         // uniform registers: 14 x 16-bit
; IMM:         // general purpose registers: 5 x 32-bit
; IMM:         mov_imm r{{[0-9]+}}, 305419896
; IMM:         mov_imm r{{[0-9]+}}, 19088743
; CHECK-LABEL: uconst:
; CHECK:         // uniform registers: 18 x 16-bit (threadgroups_per_grid at u4)
; CHECK:         // general purpose registers: 3 x 32-bit
; CHECK-NOT:     mov_imm
; CHECK:         xor r2, r2, u7
; CHECK-NEXT:    imadd r2, r2, u8, 0
; CHECK-NEXT:    xor r2, r2, u7
; Extension 6, 2 constants: half 14 = 0x12345678, half 16 = 0x01234567.
; SECTION: 06000000 02000000
; SECTION: 0e000000 78563412 10000000 67452301
define agx_kernel void @uconst(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %i
  %v = load i32, ptr addrspace(1) %p
  %x = xor i32 %v, 305419896
  %y = mul i32 %x, 19088743
  %z = xor i32 %y, 305419896
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %z, ptr addrspace(1) %q
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
