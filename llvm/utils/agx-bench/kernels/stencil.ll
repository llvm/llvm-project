; 1-D 5-point stencil with clamped edges.
; agx-cost: --uniform 4=1024 --wg 256
declare i32 @llvm.agx.thread.position.in.grid.x()
define agx_kernel void @stencil(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %last = add i32 %n, -1
  %im2 = add i32 %i, -2
  %im1 = add i32 %i, -1
  %ip1 = add i32 %i, 1
  %ip2 = add i32 %i, 2
  %c0 = call i32 @llvm.smin.i32(i32 %last, i32 %i)
  %c1a = call i32 @llvm.smax.i32(i32 %im2, i32 0)
  %c2a = call i32 @llvm.smax.i32(i32 %im1, i32 0)
  %c3a = call i32 @llvm.smin.i32(i32 %ip1, i32 %last)
  %c4a = call i32 @llvm.smin.i32(i32 %ip2, i32 %last)
  %p0 = getelementptr float, ptr addrspace(1) %in, i32 %c0
  %p1 = getelementptr float, ptr addrspace(1) %in, i32 %c1a
  %p2 = getelementptr float, ptr addrspace(1) %in, i32 %c2a
  %p3 = getelementptr float, ptr addrspace(1) %in, i32 %c3a
  %p4 = getelementptr float, ptr addrspace(1) %in, i32 %c4a
  %v0 = load float, ptr addrspace(1) %p0
  %v1 = load float, ptr addrspace(1) %p1
  %v2 = load float, ptr addrspace(1) %p2
  %v3 = load float, ptr addrspace(1) %p3
  %v4 = load float, ptr addrspace(1) %p4
  %s1 = fadd float %v1, %v4
  %s2 = fadd float %v2, %v3
  %t1 = fmul float %s1, 0.0625
  %t2 = call float @llvm.fma.f32(float %s2, float 0.25, float %t1)
  %r = call float @llvm.fma.f32(float %v0, float 0.375, float %t2)
  %in.range = icmp ult i32 %i, %n
  br i1 %in.range, label %st, label %end
st:
  %po = getelementptr float, ptr addrspace(1) %out, i32 %i
  store float %r, ptr addrspace(1) %po
  br label %end
end:
  ret void
}
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare float @llvm.fma.f32(float, float, float)
