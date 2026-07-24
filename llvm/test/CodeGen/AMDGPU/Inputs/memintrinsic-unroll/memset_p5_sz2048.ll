define void @memset_p5_sz2048(ptr addrspace(5) %dst) #1 {
entry:
  tail call void @llvm.memset.p5.i64(ptr addrspace(5) noundef nonnull %dst, i8 65, i64 2048, i1 false)
  ret void
}
attributes #1 = { nounwind }
