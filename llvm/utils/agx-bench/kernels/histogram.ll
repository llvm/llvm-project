; 256-bin histogram of bytes: threadgroup atomics, then one device atomic per bin.
; agx-cost: --uniform 4=1024 --wg 256
@bins = internal addrspace(3) global [256 x i32] undef
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare void @llvm.agx.threadgroup.barrier()
define agx_kernel void @histogram(ptr addrspace(1) %data, ptr addrspace(1) %hist, i32 %n) {
  %g = call i32 @llvm.agx.thread.position.in.grid.x()
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pb = getelementptr [256 x i32], ptr addrspace(3) @bins, i32 0, i32 %t
  store i32 0, ptr addrspace(3) %pb
  call void @llvm.agx.threadgroup.barrier()
  %in = icmp ult i32 %g, %n
  br i1 %in, label %count, label %merge
count:
  %pd = getelementptr i8, ptr addrspace(1) %data, i32 %g
  %b = load i8, ptr addrspace(1) %pd
  %bi = zext i8 %b to i32
  %pbin = getelementptr [256 x i32], ptr addrspace(3) @bins, i32 0, i32 %bi
  %o = atomicrmw add ptr addrspace(3) %pbin, i32 1 monotonic
  br label %merge
merge:
  call void @llvm.agx.threadgroup.barrier()
  %c = load i32, ptr addrspace(3) %pb
  %ph = getelementptr i32, ptr addrspace(1) %hist, i32 %t
  %o2 = atomicrmw add ptr addrspace(1) %ph, i32 %c monotonic
  ret void
}
