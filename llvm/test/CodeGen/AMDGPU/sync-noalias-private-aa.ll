; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=aa-eval -aa-pipeline=amdgpu-aa,basic-aa -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; Generic AA conservatively reports ModRef for a synchronizing operation against
; a non-byval pointer argument (see llvm/test/Transforms/GVN/fence-noalias-syncscope.ll).
; AMDGPU AA restores precision for private (scratch) memory, which is per-workitem
; and cannot be reached by peer threads, while keeping LDS/local and global memory
; conservative because those are shared.

; Private (scratch) argument: a cross-thread sync op is NoModRef.
; CHECK-LABEL: Function: private:
; CHECK:  NoModRef:  Ptr: i32* %a	<->  fence syncscope("agent") acq_rel
; CHECK:  NoModRef:  Ptr: i32* %a	<->  {{.*}}atomicrmw add ptr addrspace(5) %x
; CHECK:  NoModRef:  Ptr: i32* %a	<->  {{.*}}load atomic i32, ptr addrspace(5) %x
; CHECK:  NoModRef:  Ptr: i32* %a	<->  store atomic i32 0, ptr addrspace(5) %x release
define void @private(ptr addrspace(5) noalias %a, ptr addrspace(5) noalias %x) {
  store i32 0, ptr addrspace(5) %a
  fence syncscope("agent") acq_rel
  atomicrmw add ptr addrspace(5) %x, i32 1 acq_rel
  load atomic i32, ptr addrspace(5) %x acquire, align 4
  store atomic i32 0, ptr addrspace(5) %x release, align 4
  ret void
}

; LDS/local is workgroup-shared: a cross-thread sync op is ModRef.
; CHECK-LABEL: Function: local:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence syncscope("agent") acq_rel
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  {{.*}}atomicrmw add ptr addrspace(3) %x
define void @local(ptr addrspace(3) noalias %a, ptr addrspace(3) noalias %x) {
  store i32 0, ptr addrspace(3) %a
  fence syncscope("agent") acq_rel
  atomicrmw add ptr addrspace(3) %x, i32 1 acq_rel
  ret void
}

; Global memory is shared: a cross-thread sync op is ModRef.
; CHECK-LABEL: Function: global:
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  fence syncscope("agent") acq_rel
; CHECK:  Both ModRef:  Ptr: i32* %a	<->  {{.*}}atomicrmw add ptr addrspace(1) %x
define void @global(ptr addrspace(1) noalias %a, ptr addrspace(1) noalias %x) {
  store i32 0, ptr addrspace(1) %a
  fence syncscope("agent") acq_rel
  atomicrmw add ptr addrspace(1) %x, i32 1 acq_rel
  ret void
}

; A private pointer obtained by addrspacecast from generic is not proof of
; peer-unreachability (the underlying object is generic), so it stays ModRef.
; CHECK-LABEL: Function: cast_generic_to_private:
; CHECK:  Both ModRef:  Ptr: i32* %p	<->  fence syncscope("agent") acq_rel
; CHECK:  Both ModRef:  Ptr: i32* %p	<->  {{.*}}atomicrmw add ptr addrspace(5) %x
define void @cast_generic_to_private(ptr noalias %g, ptr addrspace(5) noalias %x) {
  %p = addrspacecast ptr %g to ptr addrspace(5)
  store i32 0, ptr addrspace(5) %p
  fence syncscope("agent") acq_rel
  atomicrmw add ptr addrspace(5) %x, i32 1 acq_rel
  ret void
}

; select of two private pointers: every underlying object is private, NoModRef.
; CHECK-LABEL: Function: select_private:
; CHECK:  NoModRef:  Ptr: i32* %sel	<->  fence syncscope("agent") acq_rel
define void @select_private(ptr addrspace(5) noalias %a, ptr addrspace(5) noalias %b, i1 %c) {
  %sel = select i1 %c, ptr addrspace(5) %a, ptr addrspace(5) %b
  store i32 0, ptr addrspace(5) %sel
  fence syncscope("agent") acq_rel
  ret void
}

; phi of two private pointers: every underlying object is private, NoModRef.
; CHECK-LABEL: Function: phi_private:
; CHECK:  NoModRef:  Ptr: i32* %p	<->  fence syncscope("agent") acq_rel
define void @phi_private(ptr addrspace(5) noalias %a, ptr addrspace(5) noalias %b, i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %j
f:
  br label %j
j:
  %p = phi ptr addrspace(5) [ %a, %t ], [ %b, %f ]
  store i32 0, ptr addrspace(5) %p
  fence syncscope("agent") acq_rel
  ret void
}

; A pointer loaded from memory is still typed private (AS5 is always scratch),
; so it remains peer-unreachable: NoModRef.
; CHECK-LABEL: Function: loaded_private_ptr:
; CHECK:  NoModRef:  Ptr: i32* %p	<->  fence syncscope("agent") acq_rel
define void @loaded_private_ptr(ptr addrspace(5) noalias %pp) {
  %p = load ptr addrspace(5), ptr addrspace(5) %pp
  store i32 0, ptr addrspace(5) %p
  fence syncscope("agent") acq_rel
  ret void
}

; byval private argument is a per-lane copy in scratch: NoModRef.
; CHECK-LABEL: Function: byval_private:
; CHECK:  NoModRef:  Ptr: i32* %a	<->  fence syncscope("agent") acq_rel
define void @byval_private(ptr addrspace(5) byval(i32) noalias %a) {
  store i32 0, ptr addrspace(5) %a
  fence syncscope("agent") acq_rel
  ret void
}
