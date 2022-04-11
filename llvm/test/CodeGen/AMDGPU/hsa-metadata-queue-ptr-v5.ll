; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=5 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK  %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 --amdhsa-code-object-version=5 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK  %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=5 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefixes=CHECK,GFX9  %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=5 < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 --amdhsa-code-object-version=5 < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=5 < %s | FileCheck --check-prefixes=CHECK,GFX9 %s


; On gfx8, the queue ptr is required for this addrspacecast.
; CHECK: - .args:
; PRE-GFX9:          .offset:         208
; PRE-GFX9-NEXT:     .size:           8
; PRE-GFX9-NEXT:     .value_kind:     hidden_queue_ptr
; GFX9-NOT:          .value_kind:     hidden_queue_ptr
; CHECK:             .name:           addrspacecast_requires_queue_ptr
; CHECK:             .symbol:         addrspacecast_requires_queue_ptr.kd
define amdgpu_kernel void @addrspacecast_requires_queue_ptr(i32 addrspace(5)* %ptr.private, i32 addrspace(3)* %ptr.local) {
  %flat.private = addrspacecast i32 addrspace(5)* %ptr.private to i32*
  %flat.local = addrspacecast i32 addrspace(3)* %ptr.local to i32*
  store volatile i32 1, i32* %flat.private
  store volatile i32 2, i32* %flat.local
  ret void
}

; CHECK: - .args:
; PRE-GFX9:          .value_kind:     hidden_multigrid_sync_arg
; PRE-GFX9-NEXT:          .offset:         200
; PRE-GFX9-NEXT:     .size:           4
; PRE-GFX9-NEXT:     .value_kind:     hidden_private_base
; PRE-GFX9-NEXT:     .offset:         204
; PRE-GFX9-NEXT:     .size:           4
; PRE-GFX9-NEXT:     .value_kind:     hidden_shared_base
; GFX9-NOT:          .value_kind:     hidden_multigrid_sync_arg
; GFX9-NOT:          .value_kind:     hidden_private_base
; CKECK-NOT:         .value_kind:     hidden_queue_ptr
; CHECK:             .name:           is_shared_requires_queue_ptr
; CHECK:             .symbol:         is_shared_requires_queue_ptr.kd
define amdgpu_kernel void @is_shared_requires_queue_ptr(i8* %ptr) {
  %is.shared = call i1 @llvm.amdgcn.is.shared(i8* %ptr)
  %zext = zext i1 %is.shared to i32
  store volatile i32 %zext, i32 addrspace(1)* undef
  ret void
}

; CHECK: - .args:
; PRE-GFX9:             .value_kind:     hidden_multigrid_sync_arg
; PRE-GFX9-NEXT:          .offset:         200
; PRE-GFX9-NEXT:     .size:           4
; PRE-GFX9-NEXT:     .value_kind:     hidden_private_base
; PRE-GFX9-NEXT:     .offset:         204
; PRE-GFX9-NEXT:     .size:           4
; PRE-GFX9-NEXT:     .value_kind:     hidden_shared_base
; GFX9-NOT:          .value_kind:     hidden_private_base
; GFX9-NOT:          .value_kind:     hidden_shared_base
; CKECK-NOT:         .value_kind:     hidden_queue_ptr
; CHECK:             .name:           is_private_requires_queue_ptr
; CHECK:             .symbol:         is_private_requires_queue_ptr.kd
define amdgpu_kernel void @is_private_requires_queue_ptr(i8* %ptr) {
  %is.private = call i1 @llvm.amdgcn.is.private(i8* %ptr)
  %zext = zext i1 %is.private to i32
  store volatile i32 %zext, i32 addrspace(1)* undef
  ret void
}

; CHECK: - .args:
; PRE-GFX9:             .value_kind:     hidden_multigrid_sync_arg
; PRE-GFX9-NEXT:          .offset:         192
; PRE-GFX9-NEXT:     .size:           4
; PRE-GFX9-NEXT:     .value_kind:     hidden_private_base
; PRE-GFX9-NEXT:     .offset:         196
; PRE-GFX9-NEXT:     .size:           4
; PRE-GFX9-NEXT:     .value_kind:     hidden_shared_base
; PRE-GFX9-NEXT:     .address_space:  global
; PRE-GFX9-NEXT:     .offset:         200
; PRE-GFX9-NEXT:     .size:           8
; PRE-GFX9-NEXT:     .value_kind:     hidden_queue_ptr
; GFX9-NOT:          .value_kind:     hidden_private_base
; GFX9-NOT:          .value_kind:     hidden_shared_base
; GFX9-NOT:          .value_kind:     hidden_queue_ptr
; CHECK:             .name:           trap_requires_queue_ptr
; CHECK:             .symbol:         trap_requires_queue_ptr.kd
define amdgpu_kernel void @trap_requires_queue_ptr() {
  call void @llvm.trap()
  unreachable
}

; CHECK: - .args:
; CHECK:             .offset:         208
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_queue_ptr
; CHECK:             .name:           amdgcn_queue_ptr_requires_queue_ptr
; CHECK:             .symbol:         amdgcn_queue_ptr_requires_queue_ptr.kd
define amdgpu_kernel void @amdgcn_queue_ptr_requires_queue_ptr(i64 addrspace(1)* %ptr)  {
  %queue.ptr = call i8 addrspace(4)* @llvm.amdgcn.queue.ptr()
  %implicitarg.ptr = call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %dispatch.ptr = call i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
  %dispatch.id = call i64 @llvm.amdgcn.dispatch.id()
  %queue.load = load volatile i8, i8 addrspace(4)* %queue.ptr
  %implicitarg.load = load volatile i8, i8 addrspace(4)* %implicitarg.ptr
  %dispatch.load = load volatile i8, i8 addrspace(4)* %dispatch.ptr
  store volatile i64 %dispatch.id, i64 addrspace(1)* %ptr
  ret void
}


declare noalias i8 addrspace(4)* @llvm.amdgcn.queue.ptr()
declare noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
declare i64 @llvm.amdgcn.dispatch.id()
declare noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
declare i1 @llvm.amdgcn.is.shared(i8*)
declare i1 @llvm.amdgcn.is.private(i8*)
declare void @llvm.trap()
declare void @llvm.debugtrap()
