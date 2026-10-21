; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: error: {{.*}}in function @signal void (): llvm.amdgcn.s.barrier.signal requires target feature 'gfx12-insts'
define amdgpu_kernel void @signal() {
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  ret void
}

; ERR: error: {{.*}}in function @signal_var void (): llvm.amdgcn.s.barrier.signal.var requires target feature 'gfx12-insts'
define amdgpu_kernel void @signal_var() {
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) null, i32 4)
  ret void
}

; ERR: error: {{.*}}in function @signal_isfirst void (): llvm.amdgcn.s.barrier.signal.isfirst requires target feature 'gfx12-insts'
define amdgpu_kernel void @signal_isfirst() {
  %r = call i1 @llvm.amdgcn.s.barrier.signal.isfirst(i32 -1)
  ret void
}

; ERR: error: {{.*}}in function @init void (): llvm.amdgcn.s.barrier.init requires target feature 'gfx12-insts'
define amdgpu_kernel void @init() {
  call void @llvm.amdgcn.s.barrier.init(ptr addrspace(15) null, i32 4)
  ret void
}

; ERR: error: {{.*}}in function @join void (): llvm.amdgcn.s.barrier.join requires target feature 'gfx12-insts'
define amdgpu_kernel void @join() {
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) null)
  ret void
}

; ERR: error: {{.*}}in function @wait void (): llvm.amdgcn.s.barrier.wait requires target feature 'gfx12-insts'
define amdgpu_kernel void @wait() {
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  ret void
}

; ERR: error: {{.*}}in function @leave void (): llvm.amdgcn.s.barrier.leave requires target feature 'gfx12-insts'
define amdgpu_kernel void @leave() {
  call void @llvm.amdgcn.s.barrier.leave(i16 -1)
  ret void
}

; ERR: error: {{.*}}in function @get_state void (): llvm.amdgcn.s.get.barrier.state requires target feature 'gfx12-insts'
define amdgpu_kernel void @get_state() {
  %r = call i32 @llvm.amdgcn.s.get.barrier.state(i32 -1)
  ret void
}

; ERR: error: {{.*}}in function @get_named_state void (): llvm.amdgcn.s.get.named.barrier.state requires target feature 'gfx12-insts'
define amdgpu_kernel void @get_named_state() {
  %r = call i32 @llvm.amdgcn.s.get.named.barrier.state(ptr addrspace(15) null)
  ret void
}
