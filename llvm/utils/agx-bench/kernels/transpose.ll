; 32x32 tile transpose through threadgroup memory (rows padded to 33 to avoid bank conflicts).
@tile = internal addrspace(3) global [32 x [33 x float]] undef
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare i32 @llvm.agx.thread.position.in.threadgroup.y()
declare i32 @llvm.agx.threadgroup.position.in.grid.x()
declare i32 @llvm.agx.threadgroup.position.in.grid.y()
declare void @llvm.agx.threadgroup.barrier()
define agx_kernel void @transpose(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %w) {
  %tx = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %ty = call i32 @llvm.agx.thread.position.in.threadgroup.y()
  %bx = call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %by = call i32 @llvm.agx.threadgroup.position.in.grid.y()
  %bx32 = shl i32 %bx, 5
  %by32 = shl i32 %by, 5
  %x = add i32 %bx32, %tx
  %y = add i32 %by32, %ty
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %pi = getelementptr float, ptr addrspace(1) %in, i32 %i
  %v = load float, ptr addrspace(1) %pi
  %ps = getelementptr [32 x [33 x float]], ptr addrspace(3) @tile, i32 0, i32 %ty, i32 %tx
  store float %v, ptr addrspace(3) %ps
  call void @llvm.agx.threadgroup.barrier()
  %pt = getelementptr [32 x [33 x float]], ptr addrspace(3) @tile, i32 0, i32 %tx, i32 %ty
  %t = load float, ptr addrspace(3) %pt
  %ox = add i32 %by32, %tx
  %oy = add i32 %bx32, %ty
  %orow = mul i32 %oy, %w
  %o = add i32 %orow, %ox
  %po = getelementptr float, ptr addrspace(1) %out, i32 %o
  store float %t, ptr addrspace(1) %po
  ret void
}
