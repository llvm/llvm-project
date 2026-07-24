define void @memmove_p1_p1_sz2048(ptr addrspace(1) align 1 %dst, ptr addrspace(1) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) noundef nonnull align 1 %dst, ptr addrspace(1) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}
attributes #1 = { nounwind }
