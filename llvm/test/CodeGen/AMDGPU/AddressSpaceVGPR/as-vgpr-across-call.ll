; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00-- -filetype=null %s 2>&1 | FileCheck %s

; The diagnostic is an error, but code is still produced: check that a
; synthesized end goes after the call rather than before it, since an end placed
; early would release the object's registers while it is still live.
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00-- -o - %s 2>/dev/null | FileCheck %s --check-prefix=ASM

; ASM-LABEL: call_in_exit_block:
; ASM: ; VGPR lifetime start: v[0:3]
; ASM: s_swappc_b64
; ASM: ; VGPR lifetime end: v[0:3]

; An object in the VGPR "as memory" address space (13) is allocated to
; caller-saved registers, so a callee is free to overwrite it. Being live across
; a call is therefore diagnosed rather than quietly reading back whatever the
; callee left behind.

declare void @extern_func()
declare void @llvm.lifetime.start.p13(ptr addrspace(13) nocapture)
declare void @llvm.lifetime.end.p13(ptr addrspace(13) nocapture)

; CHECK: error: {{.*}}in function across_call{{.*}}object in the VGPR 'as memory' address space (13) is live across a call
define void @across_call(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [4 x i32], addrspace(13)
  %p = getelementptr [4 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  call void @extern_func()
  %v = load volatile i32, ptr addrspace(13) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; The lifetime ends before the call, so the registers are free by then.
; CHECK-NOT: in function dies_before_call
define void @dies_before_call(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [4 x i32], addrspace(13)
  call void @llvm.lifetime.start.p13(ptr addrspace(13) %obj)
  %p = getelementptr [4 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  %v = load volatile i32, ptr addrspace(13) %p
  call void @llvm.lifetime.end.p13(ptr addrspace(13) %obj)
  call void @extern_func()
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; Reachability, not textual order: the object is still live on the path through
; the call.
; CHECK: error: {{.*}}in function across_call_in_branch{{.*}}live across a call
define void @across_call_in_branch(ptr addrspace(1) %out, i32 %i, i1 %c) {
entry:
  %obj = alloca [4 x i32], addrspace(13)
  call void @llvm.lifetime.start.p13(ptr addrspace(13) %obj)
  %p = getelementptr [4 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  br i1 %c, label %call, label %join

call:
  call void @extern_func()
  br label %join

join:
  %v = load volatile i32, ptr addrspace(13) %p
  call void @llvm.lifetime.end.p13(ptr addrspace(13) %obj)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; An object with no lifetime end is live to the end of the function, so a call
; in the exit block is across it too: the synthesized end goes after the call,
; not before it.
; CHECK: error: {{.*}}in function call_in_exit_block{{.*}}live across a call
define void @call_in_exit_block(ptr addrspace(1) %out, i32 %i) {
  %obj = alloca [4 x i32], addrspace(13)
  %p = getelementptr [4 x i32], ptr addrspace(13) %obj, i32 0, i32 %i
  store i32 7, ptr addrspace(13) %p
  call void @extern_func()
  ret void
}
