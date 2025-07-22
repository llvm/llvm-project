;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx1030  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx1030 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

define void @memmove_p0_p0(ptr addrspace(0) align 1 %dst, ptr addrspace(0) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p0.p0.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(0) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p0_p1(ptr addrspace(0) align 1 %dst, ptr addrspace(1) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p0.p1.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(1) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p0_p3(ptr addrspace(0) align 1 %dst, ptr addrspace(3) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p0.p3.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(3) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p0_p4(ptr addrspace(0) align 1 %dst, ptr addrspace(4) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p0.p4.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(4) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p0_p5(ptr addrspace(0) align 1 %dst, ptr addrspace(5) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p0.p5.i64(ptr addrspace(0) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}


define void @memmove_p1_p0(ptr addrspace(1) align 1 %dst, ptr addrspace(0) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p1.p0.i64(ptr addrspace(1) noundef nonnull align 1 %dst, ptr addrspace(0) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p1_p1(ptr addrspace(1) align 1 %dst, ptr addrspace(1) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) noundef nonnull align 1 %dst, ptr addrspace(1) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p1_p3(ptr addrspace(1) align 1 %dst, ptr addrspace(3) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p1.p3.i64(ptr addrspace(1) noundef nonnull align 1 %dst, ptr addrspace(3) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p1_p4(ptr addrspace(1) align 1 %dst, ptr addrspace(4) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p1.p4.i64(ptr addrspace(1) noundef nonnull align 1 %dst, ptr addrspace(4) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p1_p5(ptr addrspace(1) align 1 %dst, ptr addrspace(5) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p1.p5.i64(ptr addrspace(1) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}


define void @memmove_p3_p0(ptr addrspace(3) align 1 %dst, ptr addrspace(0) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p3.p0.i64(ptr addrspace(3) noundef nonnull align 1 %dst, ptr addrspace(0) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p3_p1(ptr addrspace(3) align 1 %dst, ptr addrspace(1) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p3.p1.i64(ptr addrspace(3) noundef nonnull align 1 %dst, ptr addrspace(1) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p3_p3(ptr addrspace(3) align 1 %dst, ptr addrspace(3) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p3.p3.i64(ptr addrspace(3) noundef nonnull align 1 %dst, ptr addrspace(3) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p3_p4(ptr addrspace(3) align 1 %dst, ptr addrspace(4) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p3.p4.i64(ptr addrspace(3) noundef nonnull align 1 %dst, ptr addrspace(4) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p3_p5(ptr addrspace(3) align 1 %dst, ptr addrspace(5) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p3.p5.i64(ptr addrspace(3) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}


define void @memmove_p5_p0(ptr addrspace(5) align 1 %dst, ptr addrspace(0) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p5.p0.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(0) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p5_p1(ptr addrspace(5) align 1 %dst, ptr addrspace(1) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p5.p1.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(1) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p5_p3(ptr addrspace(5) align 1 %dst, ptr addrspace(3) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p5.p3.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(3) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p5_p4(ptr addrspace(5) align 1 %dst, ptr addrspace(4) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p5.p4.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(4) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}

define void @memmove_p5_p5(ptr addrspace(5) align 1 %dst, ptr addrspace(5) align 1 readonly %src, i64 %sz) {
entry:
  tail call void @llvm.memmove.p5.p5.i64(ptr addrspace(5) noundef nonnull align 1 %dst, ptr addrspace(5) noundef nonnull align 1 %src, i64 %sz, i1 false)
  ret void
}


declare void @llvm.memmove.p0.p0.i64(ptr addrspace(0) nocapture writeonly, ptr addrspace(0) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p0.p1.i64(ptr addrspace(0) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p0.p3.i64(ptr addrspace(0) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p0.p4.i64(ptr addrspace(0) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p0.p5.i64(ptr addrspace(0) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p1.p0.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(0) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p1.p3.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p1.p4.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p1.p5.i64(ptr addrspace(1) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p3.p0.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(0) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p3.p1.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p3.p3.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p3.p4.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p3.p5.i64(ptr addrspace(3) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p5.p0.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(0) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p5.p1.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(1) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p5.p3.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(3) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p5.p4.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(4) nocapture readonly, i64, i1 immarg) #0
declare void @llvm.memmove.p5.p5.i64(ptr addrspace(5) nocapture writeonly, ptr addrspace(5) nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
