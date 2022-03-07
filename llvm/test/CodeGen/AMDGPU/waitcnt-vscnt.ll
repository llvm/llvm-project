; RUN: llc -march=amdgcn -mcpu=gfx802  -asm-verbose=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8,GFX8_9 %s
; RUN: llc -march=amdgcn -mcpu=gfx900  -asm-verbose=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,GFX9_10,GFX8_9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-back-off-barrier -asm-verbose=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GFX9_10 %s

; GCN-LABEL: barrier_vmcnt_global:
; GFX8:         flat_load_dword
; GFX9_10:      global_load_dword
; GFX8:         s_waitcnt vmcnt(0){{$}}
; GFX9_10:      s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:     s_barrier
define amdgpu_kernel void @barrier_vmcnt_global(i32 addrspace(1)* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp1
  %tmp4 = load i32, i32 addrspace(1)* %tmp3, align 4
  fence syncscope("singlethread") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("singlethread") acquire
  %tmp5 = add nuw nsw i64 %tmp2, 4294967296
  %tmp6 = lshr exact i64 %tmp5, 32
  %tmp7 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp6
  store i32 %tmp4, i32 addrspace(1)* %tmp7, align 4
  ret void
}

; GCN-LABEL: barrier_vscnt_global:
; GFX8:       flat_store_dword
; GFX9_10:    global_store_dword
; GFX8:       s_waitcnt vmcnt(0){{$}}
; GFX9:       s_waitcnt vmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0
; GCN-NEXT:   s_barrier
define amdgpu_kernel void @barrier_vscnt_global(i32 addrspace(1)* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = add nuw nsw i64 %tmp2, 8589934592
  %tmp4 = lshr exact i64 %tmp3, 32
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp4
  store i32 0, i32 addrspace(1)* %tmp5, align 4
  fence syncscope("singlethread") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("singlethread") acquire
  %tmp6 = add nuw nsw i64 %tmp2, 4294967296
  %tmp7 = lshr exact i64 %tmp6, 32
  %tmp8 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp7
  store i32 1, i32 addrspace(1)* %tmp8, align 4
  ret void
}

; GCN-LABEL: barrier_vmcnt_vscnt_global:
; GFX8:         flat_load_dword
; GFX9_10:      global_load_dword
; GFX8:         s_waitcnt vmcnt(0){{$}}
; GFX9_10:      s_waitcnt vmcnt(0){{$}}
; GFX10:        s_waitcnt_vscnt null, 0x0
; GCN-NEXT:     s_barrier
define amdgpu_kernel void @barrier_vmcnt_vscnt_global(i32 addrspace(1)* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = add nuw nsw i64 %tmp2, 8589934592
  %tmp4 = lshr exact i64 %tmp3, 32
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp4
  store i32 0, i32 addrspace(1)* %tmp5, align 4
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp1
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  fence syncscope("singlethread") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("singlethread") acquire
  %tmp8 = add nuw nsw i64 %tmp2, 4294967296
  %tmp9 = lshr exact i64 %tmp8, 32
  %tmp10 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp9
  store i32 %tmp7, i32 addrspace(1)* %tmp10, align 4
  ret void
}

; GCN-LABEL: barrier_vmcnt_flat:
; GCN:      flat_load_dword
; GCN:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN-NEXT: s_barrier
define amdgpu_kernel void @barrier_vmcnt_flat(i32* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = getelementptr inbounds i32, i32* %arg, i64 %tmp1
  %tmp4 = load i32, i32* %tmp3, align 4
  fence syncscope("singlethread") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("singlethread") acquire
  %tmp5 = add nuw nsw i64 %tmp2, 4294967296
  %tmp6 = lshr exact i64 %tmp5, 32
  %tmp7 = getelementptr inbounds i32, i32* %arg, i64 %tmp6
  store i32 %tmp4, i32* %tmp7, align 4
  ret void
}

; GCN-LABEL: barrier_vscnt_flat:
; GCN:         flat_store_dword
; GFX8_9:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:       s_waitcnt lgkmcnt(0){{$}}
; GFX10:       s_waitcnt_vscnt null, 0x0
; GCN-NEXT:    s_barrier
define amdgpu_kernel void @barrier_vscnt_flat(i32* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = add nuw nsw i64 %tmp2, 8589934592
  %tmp4 = lshr exact i64 %tmp3, 32
  %tmp5 = getelementptr inbounds i32, i32* %arg, i64 %tmp4
  store i32 0, i32* %tmp5, align 4
  fence syncscope("singlethread") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("singlethread") acquire
  %tmp6 = add nuw nsw i64 %tmp2, 4294967296
  %tmp7 = lshr exact i64 %tmp6, 32
  %tmp8 = getelementptr inbounds i32, i32* %arg, i64 %tmp7
  store i32 1, i32* %tmp8, align 4
  ret void
}

; GCN-LABEL: barrier_vmcnt_vscnt_flat:
; GCN:        flat_load_dword
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0
; GCN-NEXT:   s_barrier
define amdgpu_kernel void @barrier_vmcnt_vscnt_flat(i32* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = add nuw nsw i64 %tmp2, 8589934592
  %tmp4 = lshr exact i64 %tmp3, 32
  %tmp5 = getelementptr inbounds i32, i32* %arg, i64 %tmp4
  store i32 0, i32* %tmp5, align 4
  %tmp6 = getelementptr inbounds i32, i32* %arg, i64 %tmp1
  %tmp7 = load i32, i32* %tmp6, align 4
  fence syncscope("singlethread") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("singlethread") acquire
  %tmp8 = add nuw nsw i64 %tmp2, 4294967296
  %tmp9 = lshr exact i64 %tmp8, 32
  %tmp10 = getelementptr inbounds i32, i32* %arg, i64 %tmp9
  store i32 %tmp7, i32* %tmp10, align 4
  ret void
}

; GCN-LABEL: barrier_vmcnt_vscnt_flat_workgroup:
; GCN:        flat_load_dword
; GFX8_9:     s_waitcnt lgkmcnt(0){{$}}
; GFX8_9:     s_waitcnt vmcnt(0){{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0
; GCN-NEXT:   s_barrier
define amdgpu_kernel void @barrier_vmcnt_vscnt_flat_workgroup(i32* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = add nuw nsw i64 %tmp2, 8589934592
  %tmp4 = lshr exact i64 %tmp3, 32
  %tmp5 = getelementptr inbounds i32, i32* %arg, i64 %tmp4
  store i32 0, i32* %tmp5, align 4
  %tmp6 = getelementptr inbounds i32, i32* %arg, i64 %tmp1
  %tmp7 = load i32, i32* %tmp6, align 4
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %tmp8 = add nuw nsw i64 %tmp2, 4294967296
  %tmp9 = lshr exact i64 %tmp8, 32
  %tmp10 = getelementptr inbounds i32, i32* %arg, i64 %tmp9
  store i32 %tmp7, i32* %tmp10, align 4
  ret void
}

; GCN-LABEL: load_vmcnt_global:
; GFX8:     flat_load_dword
; GFX9_10:  global_load_dword
; GFX8:     s_waitcnt vmcnt(0){{$}}
; GFX9_10:  s_waitcnt vmcnt(0){{$}}
; GCN-NEXT: {{global|flat}}_store_dword
define amdgpu_kernel void @load_vmcnt_global(i32 addrspace(1)* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp1
  %tmp4 = load i32, i32 addrspace(1)* %tmp3, align 4
  %tmp5 = add nuw nsw i64 %tmp2, 4294967296
  %tmp6 = lshr exact i64 %tmp5, 32
  %tmp7 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp6
  store i32 %tmp4, i32 addrspace(1)* %tmp7, align 4
  ret void
}

; GCN-LABEL: load_vmcnt_flat:
; GCN:      flat_load_dword
; GCN-NOT:  vscnt
; GCN:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN-NEXT: {{global|flat}}_store_dword
define amdgpu_kernel void @load_vmcnt_flat(i32* %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = shl nuw nsw i64 %tmp1, 32
  %tmp3 = getelementptr inbounds i32, i32* %arg, i64 %tmp1
  %tmp4 = load i32, i32* %tmp3, align 4
  %tmp5 = add nuw nsw i64 %tmp2, 4294967296
  %tmp6 = lshr exact i64 %tmp5, 32
  %tmp7 = getelementptr inbounds i32, i32* %arg, i64 %tmp6
  store i32 %tmp4, i32* %tmp7, align 4
  ret void
}

; GCN-LABEL: store_vscnt_private:
; GCN:         buffer_store_dword
; GFX8_9:      s_waitcnt vmcnt(0)
; GFX10:       s_waitcnt_vscnt null, 0x0
; GCN-NEXT:    s_setpc_b64
define void @store_vscnt_private(i32 addrspace(5)* %p) {
  store i32 0, i32 addrspace(5)* %p
  ret void
}

; GCN-LABEL: store_vscnt_global:
; GFX8:        flat_store_dword
; GFX9_10:     global_store_dword
; GFX8_9:      s_waitcnt vmcnt(0)
; GFX10:       s_waitcnt_vscnt null, 0x0
; GCN-NEXT:    s_setpc_b64
define void @store_vscnt_global(i32 addrspace(1)* %p) {
  store i32 0, i32 addrspace(1)* %p
  ret void
}

; GCN-LABEL: store_vscnt_flat:
; GCN:         flat_store_dword
; GFX8_9:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:       s_waitcnt lgkmcnt(0){{$}}
; GFX10:       s_waitcnt_vscnt null, 0x0
; GCN-NEXT:    s_setpc_b64
define void @store_vscnt_flat(i32* %p) {
  store i32 0, i32* %p
  ret void
}

; GCN-LABEL: function_prologue:
; GCN:        s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0
; GCN-NEXT:   s_setpc_b64
define void @function_prologue() {
  ret void
}

declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.amdgcn.workitem.id.x()
