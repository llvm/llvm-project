; Block sum: threadgroup memory tree reduction of 256 floats, one atomic add per block.
@buf = internal addrspace(3) global [256 x float] undef
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare void @llvm.agx.threadgroup.barrier()
define agx_kernel void @reduce(ptr addrspace(1) %x, ptr addrspace(1) %out) {
entry:
  %g = call i32 @llvm.agx.thread.position.in.grid.x()
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %p = getelementptr float, ptr addrspace(1) %x, i32 %g
  %v = load float, ptr addrspace(1) %p
  %s = getelementptr [256 x float], ptr addrspace(3) @buf, i32 0, i32 %t
  store float %v, ptr addrspace(3) %s
  br label %loop
loop:
  %w = phi i32 [ 128, %entry ], [ %w2, %next ]
  call void @llvm.agx.threadgroup.barrier()
  %act = icmp ult i32 %t, %w
  br i1 %act, label %add, label %next
add:
  %o = add i32 %t, %w
  %so = getelementptr [256 x float], ptr addrspace(3) @buf, i32 0, i32 %o
  %a = load float, ptr addrspace(3) %s
  %b = load float, ptr addrspace(3) %so
  %c = fadd float %a, %b
  store float %c, ptr addrspace(3) %s
  br label %next
next:
  %w2 = lshr i32 %w, 1
  %more = icmp ne i32 %w2, 0
  br i1 %more, label %loop, label %done
done:
  %first = icmp eq i32 %t, 0
  br i1 %first, label %atom, label %end
atom:
  %r = load float, ptr addrspace(3) @buf
  %old = atomicrmw fadd ptr addrspace(1) %out, float %r monotonic
  br label %end
end:
  ret void
}
