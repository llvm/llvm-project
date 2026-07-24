define void @memmove_p5_p5_sz2048(ptr addrspace(5) align 1 %dst, ptr addrspace(5) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memmove.p5.p5.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}
attributes #1 = { nounwind }
