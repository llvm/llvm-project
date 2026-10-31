; RUN: opt -S -mtriple=amdgpu9.50 -passes=amdgpu-attributor %s | FileCheck %s

; The async marker intrinsics only set/await a marker in the stream of async
; requests; they cannot transfer control out of the module. Make sure they do
; not defeat the AGPR allocation inference, which conservatively gives up on
; any call that is not nocallback.

; CHECK: define amdgpu_kernel void @asyncmark_kernel({{.*}}) #[[ATTR:[0-9]+]]
define amdgpu_kernel void @asyncmark_kernel(ptr addrspace(1) %out) {
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  store i32 0, ptr addrspace(1) %out
  ret void
}

; CHECK: attributes #[[ATTR]] = {{{.*}}"amdgpu-agpr-alloc"="0"
