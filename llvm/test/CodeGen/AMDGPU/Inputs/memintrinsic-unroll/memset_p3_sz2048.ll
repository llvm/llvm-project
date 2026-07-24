define void @memset_p3_sz2048(ptr addrspace(3) %dst) #1 {
entry:
  tail call void @llvm.memset.p3.i64(ptr addrspace(3) noundef nonnull %dst, i8 65, i64 2048, i1 false)
  ret void
}
attributes #1 = { nounwind }
