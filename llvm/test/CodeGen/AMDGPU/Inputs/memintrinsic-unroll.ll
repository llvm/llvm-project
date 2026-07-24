define void @memcpy_p0_p0_sz2048(ptr addrspace(0) align 1 %dst, ptr addrspace(0) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(0) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memcpy_p1_p1_sz2048(ptr addrspace(1) align 1 %dst, ptr addrspace(1) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noundef nonnull align 1 %dst, ptr addrspace(1) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memcpy_p0_p4_sz2048(ptr addrspace(0) align 1 %dst, ptr addrspace(4) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memcpy.p0.p4.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(4) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memcpy_p5_p5_sz2048(ptr addrspace(5) align 1 %dst, ptr addrspace(5) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memcpy.p5.p5.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memcpy_p0_p5_sz2048(ptr addrspace(0) align 1 %dst, ptr addrspace(5) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memcpy.p0.p5.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memmove_p0_p0_sz2048(ptr addrspace(0) align 1 %dst, ptr addrspace(0) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memmove.p0.p0.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(0) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memmove_p1_p1_sz2048(ptr addrspace(1) align 1 %dst, ptr addrspace(1) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) noundef nonnull align 1 %dst, ptr addrspace(1) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memmove_p0_p4_sz2048(ptr addrspace(0) align 1 %dst, ptr addrspace(4) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memmove.p0.p4.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(4) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memmove_p5_p5_sz2048(ptr addrspace(5) align 1 %dst, ptr addrspace(5) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memmove.p5.p5.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memmove_p0_p5_sz2048(ptr addrspace(0) align 1 %dst, ptr addrspace(5) align 1 readonly %src) #1 {
entry:
  tail call void @llvm.memmove.p0.p5.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 2048, i1 false)
  ret void
}

define void @memset_p0_sz2048(ptr addrspace(0) %dst) #1 {
entry:
  tail call void @llvm.memset.p0.i64(ptr addrspace(0) noundef nonnull %dst, i8 65, i64 2048, i1 false)
  ret void
}

define void @memset_p1_sz2048(ptr addrspace(1) %dst) #1 {
entry:
  tail call void @llvm.memset.p1.i64(ptr addrspace(1) noundef nonnull %dst, i8 65, i64 2048, i1 false)
  ret void
}

define void @memset_p3_sz2048(ptr addrspace(3) %dst) #1 {
entry:
  tail call void @llvm.memset.p3.i64(ptr addrspace(3) noundef nonnull %dst, i8 65, i64 2048, i1 false)
  ret void
}

define void @memset_p5_sz2048(ptr addrspace(5) %dst) #1 {
entry:
  tail call void @llvm.memset.p5.i64(ptr addrspace(5) noundef nonnull %dst, i8 65, i64 2048, i1 false)
  ret void
}

attributes #1 = { nounwind }
