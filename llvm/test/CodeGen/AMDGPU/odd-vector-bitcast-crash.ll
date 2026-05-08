; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -O2 %s -o /dev/null

target triple = "amdgcn-amd-amdhsa"

define void @repro(ptr %p, <3 x i1> %mask) {
  %x = call <3 x i31> @llvm.masked.load.v3i31.p0(
    ptr %p, <3 x i1> %mask, <3 x i31> zeroinitializer)

  call void @llvm.masked.store.v3i31.p0(
    <3 x i31> %x, ptr null, <3 x i1> %mask)

  ret void
}

declare <3 x i31> @llvm.masked.load.v3i31.p0(
  ptr, <3 x i1>, <3 x i31>)

declare void @llvm.masked.store.v3i31.p0(
  <3 x i31>, ptr, <3 x i1>)
