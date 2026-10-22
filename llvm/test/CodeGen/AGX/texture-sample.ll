; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -show-mc-encoding < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-strict -filetype=obj -o /dev/null < %s

; llvm.agx.texture.sample is a bindless texture_sample: the descriptor at the
; texture heap address (u0_u1, which every kernel of the module receives
; before its arguments) plus the offset register, a register sampler index
; into the command's sampler heap, automatic LOD, the four channels as one
; register tuple (not overlapping the sources) and a wait before they are
; read. The kernel record gets extension 5 (texture heap at half 0). The
; encodings are the forms Mesa's packer emits (agx-strict accepts them).

declare <4 x i32> @llvm.agx.texture.sample(i32, i16, float, float, i32)
declare i32 @llvm.agx.thread.position.in.grid.x()

; CHECK-LABEL: sample2d:
; CHECK: // uniform registers: 18 x 16-bit (threadgroups_per_grid at u6)
; CHECK: // texture heap at u0_u1
; CHECK: texture_sample 0, r7_r8_r9_r10, r4, r3l, r5_r6, 2, xyzw // encoding: [0x31,0x1d,0x0a,0x00,0xc8,0x62,0x0f,0x46]
; CHECK-NEXT: wait 0
; CHECK-NEXT: device_store 0, i32, xyzw, r7_r8_r9_r10, u2_u3, r1, signed, lsl 2
define agx_kernel void @sample2d(ptr addrspace(1) %out, i32 %off, i16 %s) {
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %u = uitofp i32 %t to float
  %x = fmul float %u, 0x3FB0000000000000
  %r = call <4 x i32> @llvm.agx.texture.sample(i32 %off, i16 %s, float %x, float 0.5, i32 2)
  %p = getelementptr <4 x i32>, ptr addrspace(1) %out, i32 %t
  store <4 x i32> %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: sample1d:
; CHECK: texture_sample 0, {{r[0-9]+_r[0-9]+_r[0-9]+_r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+[lh]}}, {{r[0-9]+}}, 0, xyzw
define agx_kernel void @sample1d(ptr addrspace(1) %out, i32 %off, i16 %s, float %x) {
  %r = call <4 x i32> @llvm.agx.texture.sample(i32 %off, i16 %s, float %x, float poison, i32 0)
  store <4 x i32> %r, ptr addrspace(1) %out
  ret void
}

; A kernel that does not sample still keeps u0_u1 for the texture heap: a
; function of its module may sample.
; CHECK-LABEL: nosample:
; CHECK: // uniform registers: 14 x 16-bit (threadgroups_per_grid at u4)
; CHECK: // texture heap at u0_u1
; CHECK: device_store 0, i32, x, {{r[0-9]+}}, u2_u3, 0, signed
define agx_kernel void @nosample(ptr addrspace(1) %out) {
  store i32 7, ptr addrspace(1) %out
  ret void
}

; The record of @sample2d: ... then extension 5, half 0.
; CHECK: .section .agx.kernels
; CHECK: .ascii "sample2d"
; CHECK: .long 5
; CHECK-NEXT: .long 0
