; RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s
; RUN: llc < %s -mtriple=r600 -mcpu=cayman | FileCheck %s

; CHECK: {{^}}test:
; CHECK: BIT_ALIGN_INT T{{[0-9]}}.X
; CHECK: BIT_ALIGN_INT T{{[0-9]}}.Y
; CHECK: BIT_ALIGN_INT T{{[0-9]}}.Z
; CHECK: BIT_ALIGN_INT * T{{[0-9]}}.W

define amdgpu_kernel void @test(ptr addrspace(1) %out, i32 %x_arg, i32 %y_arg, i32 %z_arg, i32 %w_arg, i32 %e) {
entry:
  %shl = sub i32 32, %e
  %x = add i32 %x_arg, 1
  %x.0 = shl i32 %x, %shl
  %x.1 = lshr i32 %x, %e
  %x.2 = or i32 %x.0, %x.1
  %y = add i32 %y_arg, 1
  %y.0 = shl i32 %y, %shl
  %y.1 = lshr i32 %y, %e
  %y.2 = or i32 %y.0, %y.1
  %z = add i32 %z_arg, 1
  %z.0 = shl i32 %z, %shl
  %z.1 = lshr i32 %z, %e
  %z.2 = or i32 %z.0, %z.1
  %w = add i32 %w_arg, 1
  %w.0 = shl i32 %w, %shl
  %w.1 = lshr i32 %w, %e
  %w.2 = or i32 %w.0, %w.1
  %xy = or i32 %x.2, %y.2
  %zw = or i32 %z.2, %w.2
  %xyzw = or i32 %xy, %zw
  store i32 %xyzw, ptr addrspace(1) %out
  ret void
}
