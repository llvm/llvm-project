; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-enable-object-linking -passes=amdgpu-lower-exec-sync,amdgpu-lower-module-lds < %s | FileCheck %s

; Verify that with object linking enabled:
; 1. AMDGPULowerExecSync externalizes named barriers and emits
;    amdgpu.named_barrier.uses metadata with (barrier, func...) format
; 2. AMDGPULowerModuleLDS does not handle named barriers at all
; 3. amdgpu.lds.uses does NOT contain barrier entries

@bar = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
@lds = internal addrspace(3) global [4 x i32] poison, align 4

; Internal named barrier becomes external with a module-unique hash suffix.
; CHECK: @[[BAR:__amdgpu_named_barrier\.bar\.[a-f0-9]+]] = external dso_local addrspace(3) global target("amdgcn.named.barrier", 0)
; CHECK-NOT: !absolute_symbol
; Regular LDS is packed into the per-function struct (external, for linker).
; CHECK: @__amdgpu_lds.kernel = external dso_local addrspace(3) global %__amdgpu_lds.kernel.t, align 16

define amdgpu_kernel void @kernel(i32 %idx) {
; CHECK-LABEL: define amdgpu_kernel void @kernel(
; CHECK:         call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @[[BAR]], i32 3)
; CHECK:         call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @[[BAR]])
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @bar, i32 3)
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @bar)
  call void @llvm.amdgcn.s.barrier.wait(i16 1)
  %gep = getelementptr [4 x i32], ptr addrspace(3) @lds, i32 0, i32 %idx
  store i32 42, ptr addrspace(3) %gep, align 4
  ret void
}

; Named barrier metadata: (barrier_sym, func1, ...) -- emitted by ExecSync.
; CHECK-DAG: !amdgpu.named_barrier.uses = !{[[BAR_MD:![0-9]+]]}
; CHECK-DAG: [[BAR_MD]] = !{ptr addrspace(3) @[[BAR]], ptr @kernel}
; LDS metadata must have exactly one entry (the LDS struct), no barrier entries.
; CHECK-DAG: !amdgpu.lds.uses = !{[[LDS_MD:![0-9]+]]}
; CHECK-DAG: [[LDS_MD]] = !{ptr @kernel, ptr addrspace(3) @__amdgpu_lds.kernel}
