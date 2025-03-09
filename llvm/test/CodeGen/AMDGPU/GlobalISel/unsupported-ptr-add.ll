; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o - < %s 2>&1 | FileCheck -check-prefix=GISEL-ERR %s

; GISEL-ERR: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(p8) = G_PTR_ADD %{{[0-9]+}}:_, %{{[0-9]+}}:_(i128)


define float @gep_on_rsrc(ptr addrspace(8) %rsrc) {
body:
  %next = getelementptr float, ptr addrspace(8) %rsrc, i128 1
  %res = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %next, i32 0, i32 0, i32 0)
  ret float %res
}

declare float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8), i32, i32, i32 immarg)

