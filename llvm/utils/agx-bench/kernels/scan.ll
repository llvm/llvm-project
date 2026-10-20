; Inclusive SIMD-group prefix sum of 32-bit integers (exclusive scan plus the own value).
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.simd.prefix.i32(i32, i32 immarg)
define agx_kernel void @scan(ptr addrspace(1) %x) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr i32, ptr addrspace(1) %x, i32 %i
  %v = load i32, ptr addrspace(1) %p
  %e = call i32 @llvm.agx.simd.prefix.i32(i32 %v, i32 16)
  %s = add i32 %e, %v
  store i32 %s, ptr addrspace(1) %p
  ret void
}
