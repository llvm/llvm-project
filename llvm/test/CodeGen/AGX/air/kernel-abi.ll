; An AIR kernel, as the AIR target's portable passes leave it, becomes an AGX
; kernel: its buffers are its arguments in location order, the constant
; buffers holding scalars are passed by reference (so their bytes are uniform
; registers), system values are special register reads, thread memory is
; private and the module takes AGX's triple and data layout.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %t.air.ll | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %t.air.ll -o - | FileCheck %s --check-prefix=ASM

; CHECK: target datalayout = "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32
; CHECK: target triple = "agx"
; CHECK-LABEL: define agx_kernel void @saxpy(
; CHECK-SAME: ptr addrspace(4) noalias noundef nonnull readonly byref(float) align 4 dereferenceable(4) %alpha,
; CHECK-SAME: ptr addrspace(1) %x, ptr addrspace(1) %y,
; CHECK-SAME: ptr addrspace(4) noalias noundef nonnull readonly byref(i32) align 4 dereferenceable(4) %n)
; CHECK-NEXT: entry:
; CHECK-NEXT: call i32 @llvm.agx.thread.position.in.grid.x()
; CHECK: call i32 @llvm.agx.thread.index.in.simdgroup()
; CHECK: %tmp = alloca [4 x float], align 4, addrspace(5)
; CHECK: store float 1.000000e+00, ptr addrspace(5)
; CHECK-NOT: !air.
; CHECK-NOT: !llvm.module.flags

; alpha and n are uniform registers: nothing loads them.
; ASM-LABEL: saxpy:
; ASM: uniform registers: 18 x 16-bit
; ASM: if_icmp {{.*}}, slt, r{{[0-9]+}}, u5
; ASM: device_load 0, i32, x, {{.*}}, u1_u2,
; ASM: device_load 1, i32, x, {{.*}}, u3_u4,
; ASM: fmul32 {{.*}}, u0,
; ASM: device_store {{.*}}, u3_u4,

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
declare i32 @air.thread_index_in_simdgroup()

define void @saxpy(float %alpha, ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) "air-kernel" {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %i = extractvalue { i32, i32, i32 } %t, 0
  %lane = call i32 @air.thread_index_in_simdgroup()
  %tmp = alloca [4 x float]
  %slot = getelementptr [4 x float], ptr %tmp, i32 0, i32 %lane
  store float 1.0, ptr %slot
  %ok = icmp slt i32 %i, %n
  br i1 %ok, label %body, label %done
body:
  %px = getelementptr float, ptr addrspace(1) %x, i32 %i
  %py = getelementptr float, ptr addrspace(1) %y, i32 %i
  %vx = load float, ptr addrspace(1) %px
  %vy = load float, ptr addrspace(1) %py
  %s = load float, ptr %slot
  %ax = fmul float %alpha, %vx
  %r0 = fadd float %ax, %vy
  %r = fmul float %r0, %s
  store float %r, ptr addrspace(1) %py
  br label %done
done:
  ret void
}
