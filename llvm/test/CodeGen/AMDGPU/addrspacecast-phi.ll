; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 -stop-before=phi-node-elimination -verify-machineinstrs %s -o - | FileCheck %s

; AMDGPU overrides shouldSinkAddrSpaceCastForPHI to keep a noop addrspacecast
; in a block that dominates the phi. Sinking it into the predecessors makes
; isel build the pointer phi with an undef incoming on the divergent skip
; edge, and a byte store then uses the other buffer's base
; (ROCm/rocm-systems#11305).
;
; CHECK-LABEL: name: phi_addrspacecast
; CHECK-NOT: PHI undef
; CHECK: = PHI {{%[0-9]+}}, %bb.{{[0-9]+}}, {{%[0-9]+}}, %bb.{{[0-9]+}}
; CHECK: GLOBAL_STORE_BYTE

define amdgpu_kernel void @phi_addrspacecast(ptr addrspace(4) %ka) {
entry:
  %mm.p = getelementptr inbounds i8, ptr addrspace(4) %ka, i64 0
  %bc.p = getelementptr inbounds i8, ptr addrspace(4) %ka, i64 8
  %mm.flat = load ptr, ptr addrspace(4) %mm.p, align 8
  %bc.flat = load ptr, ptr addrspace(4) %bc.p, align 8
  %mm = addrspacecast ptr %mm.flat to ptr addrspace(1)
  %bc = addrspacecast ptr %bc.flat to ptr addrspace(1)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cond = icmp eq i32 %tid, 0
  br i1 %cond, label %oob, label %in

oob:
  br label %join

in:
  br label %join

join:
  %p = phi ptr addrspace(1) [ %mm, %oob ], [ %bc, %in ]
  store i8 1, ptr addrspace(1) %p, align 1
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
