define void @memset_p1_sz2048(ptr addrspace(1) %dst) #1 {
entry:
  tail call void @llvm.memset.p1.i64(ptr addrspace(1) noundef nonnull %dst, i8 65, i64 2048, i1 false)
  ret void
}
attributes #1 = { nounwind }
