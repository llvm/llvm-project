; exp(x - max) / sum over a SIMD-group of 32 values (one row of a softmax).
declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.agx.simd.reduce.f32(float, i32 immarg)
define agx_kernel void @softmax_row(ptr addrspace(1) %x) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr float, ptr addrspace(1) %x, i32 %i
  %v = load float, ptr addrspace(1) %p
  %m = call float @llvm.agx.simd.reduce.f32(float %v, i32 7)
  %d = fsub float %v, %m
  %e = call float @llvm.exp.f32(float %d)
  %s = call float @llvm.agx.simd.reduce.f32(float %e, i32 1)
  %r = fdiv float %e, %s
  store float %r, ptr addrspace(1) %p
  ret void
}
declare float @llvm.exp.f32(float)
