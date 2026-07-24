define void @memcpy_p0_p0_sz2048(ptr addrspace(0) align 1 %dst, ptr addrspace(0) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(0) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}
attributes #1 = { nounwind }
