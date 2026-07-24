define void @memset_p0_sz2048(ptr addrspace(0) %dst) #1 {
entry:
  tail call void @llvm.memset.p0.i64(ptr addrspace(0) noundef nonnull %dst, i8 65, i64 2048, i1 false)
  ret void
}
attributes #1 = { nounwind }
