; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s

; simd_shuffle ORs the low two index bits of the threads of each quad, and
; reads the index of inactive threads too. Divergent indices are split into
; the four quad positions, and register indices are passed in r0h, which is
; zero whenever no shuffle is executing.

declare i32 @llvm.agx.thread.index.in.simdgroup()
declare float @llvm.agx.simd.shuffle.f32(float, i32)
declare float @llvm.agx.simd.shuffle.down.f32(float, i32)
declare float @llvm.agx.quad.shuffle.f32(float, i32)

; CHECK-LABEL: divergent:
; CHECK:         mov_imm r0h, 0
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NEXT:    mov_imm r0h, 0
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NEXT:    mov_imm r0h, 0
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NEXT:    mov_imm r0h, 0
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NEXT:    mov_imm r0h, 0
; CHECK-NOT:     simd_shuffle
; CHECK:         stop
define agx_kernel void @divergent(ptr addrspace(1) %out, float %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %idx = mul i32 %lane, 7
  %r = call float @llvm.agx.simd.shuffle.f32(float %x, i32 %idx)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %p
  ret void
}

; A uniform index is used directly.
; CHECK-LABEL: uniform:
; CHECK-NOT:     r0h
; CHECK:         simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, u3l
; CHECK-NOT:     simd_shuffle
; CHECK:         stop
define agx_kernel void @uniform(ptr addrspace(1) %out, float %x, i32 %i) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %v = uitofp i32 %lane to float
  %r = call float @llvm.agx.simd.shuffle.f32(float %v, i32 %i)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %r, ptr addrspace(1) %p
  ret void
}

; A divergent delta reads the computed thread; quad shuffles select between
; the four immediate positions.
; CHECK-LABEL: relative:
; CHECK-COUNT-4: simd_shuffle {{r[0-9]+}}, {{[ru][0-9]+}}, r0h
; CHECK-NOT:     simd_shuffle_down
; CHECK:         quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 0
; CHECK:         quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 1
; CHECK:         quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 2
; CHECK:         quad_shuffle {{r[0-9]+}}, {{r[0-9]+}}, 3
; CHECK:         stop
define agx_kernel void @relative(ptr addrspace(1) %out, float %x) {
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %d = and i32 %lane, 3
  %v = uitofp i32 %lane to float
  %a = call float @llvm.agx.simd.shuffle.down.f32(float %v, i32 %d)
  %b = call float @llvm.agx.quad.shuffle.f32(float %a, i32 %d)
  %p = getelementptr float, ptr addrspace(1) %out, i32 %lane
  store float %b, ptr addrspace(1) %p
  ret void
}

; Without the merge, divergent indices are used directly.
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=-quad-shuffle-index-merge < %s | FileCheck %s --check-prefix=NOMERGE
; NOMERGE-LABEL: divergent:
; NOMERGE-COUNT-1: simd_shuffle
; NOMERGE-NOT:   simd_shuffle
; NOMERGE:       stop
