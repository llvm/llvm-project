; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -O2 -tail-dup-size=1000 -tail-dup-placement-threshold=1000 -enable-tail-merge=0 < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; Need to to trigger tail duplication this during
; MachineBlockPlacement, since calls aren't tail duplicated pre-RA.

declare void @nonconvergent_func() nounwind
declare void @convergent_func() nounwind convergent
declare void @llvm.amdgcn.s.barrier() nounwind convergent
declare void @llvm.amdgcn.ds.gws.init(i32, i32) convergent inaccessiblememonly nounwind
declare void @llvm.amdgcn.ds.gws.barrier(i32, i32) convergent inaccessiblememonly nounwind
declare void @llvm.amdgcn.ds.gws.sema.release.all(i32 %offset) convergent inaccessiblememonly nounwind

; barrier shouldn't be duplicated.

; GCN-LABEL: {{^}}taildup_barrier:
; GCN: s_barrier
; GCN-NOT: s_barrier
define void @taildup_barrier(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond) nounwind {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, ptr addrspace(1) %a
  br label %call

bb2:
  store i32 1, ptr addrspace(1) %a
  br label %call

call:
  call void @llvm.amdgcn.s.barrier()
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}taildup_convergent_call:
; GCN: s_swappc_b64
; GCN-NOT: s_swappc_b64
define void @taildup_convergent_call(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond) nounwind convergent {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, ptr addrspace(1) %a
  br label %call

bb2:
  store i32 1, ptr addrspace(1) %a
  br label %call

call:
  call void @convergent_func()
  br label %ret

ret:
  ret void
}

; TODO: Currently there is only one convergent call pseudo, but this
; theoretically could use a nonconvergent variant.
; GCN-LABEL: {{^}}taildup_nonconvergent_call:
; GCN: s_swappc_b64
; GCN-NOT: s_swappc_b64
define void @taildup_nonconvergent_call(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond) nounwind convergent {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, ptr addrspace(1) %a
  br label %call

bb2:
  store i32 1, ptr addrspace(1) %a
  br label %call

call:
  call void @nonconvergent_func()
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}taildup_convergent_tailcall:
; GCN: s_setpc_b64
; GCN-NOT: s_setpc_b64
define void @taildup_convergent_tailcall(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond) nounwind convergent {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, ptr addrspace(1) %a
  br label %call

bb2:
  store i32 1, ptr addrspace(1) %a
  br label %call

call:
  tail call void @convergent_func()
  ret void
}

; GCN-LABEL: {{^}}taildup_gws_init:
; GCN: ds_gws_init
; GCN-NOT: ds_gws_init
define amdgpu_kernel void @taildup_gws_init(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond, i32 %val, i32 %offset) nounwind {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, ptr addrspace(1) %a
  br label %call

bb2:
  store i32 1, ptr addrspace(1) %a
  br label %call

call:
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 %offset)
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}taildup_gws_barrier:
; GCN: ds_gws_barrier
; GCN-NOT: ds_gws_barrier
define amdgpu_kernel void @taildup_gws_barrier(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond, i32 %val, i32 %offset) nounwind {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, ptr addrspace(1) %a
  br label %call

bb2:
  store i32 1, ptr addrspace(1) %a
  br label %call

call:
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 %offset)
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}taildup_gws_sema_release_all:
; GCN: ds_gws_sema_release_all
; GCN-NOT: ds_gws
define amdgpu_kernel void @taildup_gws_sema_release_all(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond, i32 %offset) nounwind {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, ptr addrspace(1) %a
  br label %call

bb2:
  store i32 1, ptr addrspace(1) %a
  br label %call

call:
  call void @llvm.amdgcn.ds.gws.sema.release.all(i32 %offset)
  br label %ret

ret:
  ret void
}
