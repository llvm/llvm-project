; RUN: opt -S -verify-each -passes='function(mem2reg,spirv-legalize-resource-pointer-phis)' -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s

; mem2reg replaces each load of the never-stored %slot with undef, as the
; backend's own register promotion does for a path without a store.

%struct.S = type { i32, i32 }

@.str = private unnamed_addr constant [3 x i8] c"In\00"
@.str.1 = private unnamed_addr constant [4 x i8] c"Out\00"
@.str.2 = private unnamed_addr constant [4 x i8] c"Buf\00"

; A load through a phi of two getpointer calls on one handle should load
; through one getpointer on a phi of their indices.
; CHECK-LABEL: define i32 @phi_load(
; CHECK:       a:
; CHECK-NEXT:    br label %merge
; CHECK:       b:
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    %p.idx = phi i32 [ 0, %a ], [ 1, %b ]
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 %p.idx)
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %p, align 4
define i32 @phi_load(i1 %c) {
entry:
  %h = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
}

; A store through a phi of two getpointer calls on one handle should store
; through one getpointer on a phi of their indices.
; CHECK-LABEL: define void @phi_store(
; CHECK:       merge:
; CHECK-NEXT:    %p.idx = phi i32 [ 0, %a ], [ 1, %b ]
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 %p.idx)
; CHECK-NEXT:    store i32 %v, ptr addrspace(11) %p, align 4
define void @phi_store(i1 %c, i32 %v) {
entry:
  %h = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  store i32 %v, ptr addrspace(11) %p, align 4
  ret void
}

; Rewrite both selects and keep the getpointer that another load still uses.
; The legalization is required on optnone functions too.
; CHECK-LABEL: define i32 @select_load(
; CHECK:         %h = call
; CHECK-NEXT:    %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 2)
; CHECK-NEXT:    %inner.idx = select i1 %c, i32 2, i32 3
; CHECK-NEXT:    %inner = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 %inner.idx)
; CHECK-NEXT:    %outer.idx = select i1 %d, i32 %inner.idx, i32 2
; CHECK-NEXT:    %outer = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 %outer.idx)
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %outer, align 4
; CHECK-NEXT:    %w = load i32, ptr addrspace(11) %inner, align 4
; CHECK-NEXT:    %x = load i32, ptr addrspace(11) %p0, align 4
define i32 @select_load(i1 %c, i1 %d) noinline optnone {
entry:
  %h = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 2)
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 3)
  %inner = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1
  %outer = select i1 %d, ptr addrspace(11) %inner, ptr addrspace(11) %p0
  %v = load i32, ptr addrspace(11) %outer, align 4
  %w = load i32, ptr addrspace(11) %inner, align 4
  %x = load i32, ptr addrspace(11) %p0, align 4
  %sum = add i32 %v, %w
  %result = add i32 %sum, %x
  ret i32 %result
}

; The undef entry value of a loop phi becomes index 0.
; CHECK-LABEL: define i32 @loop_phi(
; CHECK:       header:
; CHECK-NEXT:    %q.idx = phi i32 [ 0, %entry ], [ %m.idx, %body ]
; CHECK-NEXT:    %i = phi i32
; CHECK:       merge:
; CHECK-NEXT:    %m.idx = phi i32 [ %q.idx, %header ], [ 0, %a ]
; CHECK-NEXT:    %m = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 %m.idx)
; CHECK:       body:
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %m, align 4
define i32 @loop_phi(i32 %n) {
entry:
  %slot = alloca ptr addrspace(11)
  %u = load ptr addrspace(11), ptr %slot
  %h = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  br label %header
header:
  %q = phi ptr addrspace(11) [ %u, %entry ], [ %m, %body ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
  %s = phi i32 [ 0, %entry ], [ %s.next, %body ]
  %c = icmp eq i32 %i, 0
  br i1 %c, label %a, label %merge
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 0)
  br label %merge
merge:
  %m = phi ptr addrspace(11) [ %q, %header ], [ %p0, %a ]
  %done = icmp eq i32 %i, %n
  br i1 %done, label %exit, label %body
body:
  %v = load i32, ptr addrspace(11) %m, align 4
  %s.next = add i32 %s, %v
  %i.next = add i32 %i, 1
  br label %header
exit:
  ret i32 %s
}

; A loop phi whose handle is defined in the phi's block should stay unchanged,
; since the backedge brings a pointer from the previous iteration's handle.
; CHECK-LABEL: define i32 @loop_carried_pointer_changing_handle(
; CHECK:       header:
; CHECK-NEXT:    %h = phi target("spirv.VulkanBuffer", [0 x i32], 12, 1) [ %h0, %entry ], [ %h1, %back ]
; CHECK-NEXT:    %p = phi ptr addrspace(11) [ poison, %entry ], [ %prior, %back ]
; CHECK:       back:
; CHECK-NEXT:    %prior = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 2)
; CHECK:       read:
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %p, align 4
define i32 @loop_carried_pointer_changing_handle() {
entry:
  %h0 = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %h1 = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  br label %header
header:
  %h = phi target("spirv.VulkanBuffer", [0 x i32], 12, 1) [ %h0, %entry ], [ %h1, %back ]
  %p = phi ptr addrspace(11) [ poison, %entry ], [ %prior, %back ]
  %first = phi i1 [ true, %entry ], [ false, %back ]
  br i1 %first, label %back, label %read
back:
  %prior = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 2)
  br label %header
read:
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
}

; A load of a struct member through a GEP of a phi of getpointer calls should
; load through the same GEP of one getpointer on a phi of the indices.
; CHECK-LABEL: define i32 @gep_of_phi_load(
; CHECK:       merge:
; CHECK-NEXT:    %p.idx = phi i32 [ 0, %a ], [ 1, %b ]
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %h, i32 %p.idx)
; CHECK-NEXT:    %m = getelementptr inbounds nuw i8, ptr addrspace(11) %p, i64 4
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %m, align 4
define i32 @gep_of_phi_load(i1 %c) {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t.i32(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %h, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t.i32(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %h, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %m = getelementptr inbounds nuw i8, ptr addrspace(11) %p, i64 4
  %v = load i32, ptr addrspace(11) %m, align 4
  ret i32 %v
}

; A store to a struct member through a GEP of a select of getpointer calls
; should store through the same GEP of one getpointer on a select of the
; indices.
; CHECK-LABEL: define void @gep_of_select_store(
; CHECK:         %h = call
; CHECK-NEXT:    %p.idx = select i1 %c, i32 %i, i32 %j
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %h, i32 %p.idx)
; CHECK-NEXT:    %m = getelementptr inbounds nuw i8, ptr addrspace(11) %p, i64 4
; CHECK-NEXT:    store i32 %v, ptr addrspace(11) %m, align 4
define void @gep_of_select_store(i1 %c, i32 %i, i32 %j, i32 %v) {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t.i32(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %h, i32 %i)
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t.i32(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %h, i32 %j)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1
  %m = getelementptr inbounds nuw i8, ptr addrspace(11) %p, i64 4
  store i32 %v, ptr addrspace(11) %m, align 4
  ret void
}

; An atomicrmw through a phi of getpointer calls, as InterlockedAdd in each
; branch becomes after SimplifyCFG, should use one getpointer on a phi of the
; indices.
; CHECK-LABEL: define i32 @phi_atomicrmw(
; CHECK:       merge:
; CHECK-NEXT:    %p.idx = phi i32 [ 0, %a ], [ 1, %b ]
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 %p.idx)
; CHECK-NEXT:    %r = atomicrmw add ptr addrspace(11) %p, i32 1 syncscope("device") monotonic, align 4
define i32 @phi_atomicrmw(i1 %c) {
entry:
  %h = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %r = atomicrmw add ptr addrspace(11) %p, i32 1 syncscope("device") monotonic, align 4
  ret i32 %r
}

; A cmpxchg through a select of getpointer calls should use one getpointer on
; a select of the indices.
; CHECK-LABEL: define i32 @select_cmpxchg(
; CHECK:         %h = call
; CHECK-NEXT:    %p.idx = select i1 %c, i32 2, i32 3
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 %p.idx)
; CHECK-NEXT:    %r = cmpxchg ptr addrspace(11) %p, i32 0, i32 1 syncscope("device") monotonic monotonic, align 4
define i32 @select_cmpxchg(i1 %c) {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 2)
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 3)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1
  %r = cmpxchg ptr addrspace(11) %p, i32 0, i32 1 syncscope("device") monotonic monotonic, align 4
  %v = extractvalue { i32, i1 } %r, 0
  ret i32 %v
}

; A phi of getpointer calls on two different handles should stay unchanged.
; CHECK-LABEL: define i32 @two_handles(
; CHECK:         %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
define i32 @two_handles(i1 %c) {
entry:
  %h0 = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %h1 = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h0, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h1, i32 0)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
}

; A phi with a GEP of a getpointer call as one input should stay unchanged.
; CHECK-LABEL: define i32 @gep_input(
; CHECK:         %p = phi ptr addrspace(11) [ %g, %a ], [ %p1, %b ]
define i32 @gep_input(i1 %c) {
entry:
  %h = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 0)
  %g = getelementptr i8, ptr addrspace(11) %p0, i32 4
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %g, %a ], [ %p1, %b ]
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
}

; A phi of two null pointers, without a getpointer call, should stay unchanged.
; CHECK-LABEL: define i32 @null_inputs(
; CHECK:         %p = phi ptr addrspace(11) [ null, %a ], [ null, %entry ]
define i32 @null_inputs(i1 %c) {
entry:
  br i1 %c, label %a, label %merge
a:
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ null, %a ], [ null, %entry ]
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
}

; A select without a getpointer input should stay unchanged.
; CHECK-LABEL: define void @no_getpointer(
; CHECK:         %p = select i1 %c, ptr addrspace(11) poison, ptr addrspace(11) poison
; CHECK:         store i32 0, ptr addrspace(11) %p, align 4
define void @no_getpointer(i1 %c, i1 %access) {
entry:
  %p = select i1 %c, ptr addrspace(11) poison, ptr addrspace(11) poison
  br i1 %access, label %write, label %exit
write:
  store i32 0, ptr addrspace(11) %p, align 4
  br label %exit
exit:
  ret void
}

; A select of constant-buffer members should stay unchanged, since a member
; index must stay constant.
; CHECK-LABEL: define i32 @cbuffer_member_select(
; CHECK:         %p0 = call ptr addrspace(12) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", %struct.S, 2, 0) %h, i32 0)
; CHECK-NEXT:    %p1 = call ptr addrspace(12) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", %struct.S, 2, 0) %h, i32 1)
; CHECK-NEXT:    %p = select i1 %c, ptr addrspace(12) %p0, ptr addrspace(12) %p1
; CHECK-NEXT:    %v = load i32, ptr addrspace(12) %p, align 4
define i32 @cbuffer_member_select(i1 %c) {
entry:
  %h = call target("spirv.VulkanBuffer", %struct.S, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_struct.Ss_2_0t(i32 0, i32 3, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(12) @llvm.spv.resource.getpointer.p12.tspirv.VulkanBuffer_s_struct.Ss_2_0t.i32(target("spirv.VulkanBuffer", %struct.S, 2, 0) %h, i32 0)
  %p1 = call ptr addrspace(12) @llvm.spv.resource.getpointer.p12.tspirv.VulkanBuffer_s_struct.Ss_2_0t.i32(target("spirv.VulkanBuffer", %struct.S, 2, 0) %h, i32 1)
  %p = select i1 %c, ptr addrspace(12) %p0, ptr addrspace(12) %p1
  %v = load i32, ptr addrspace(12) %p, align 4
  ret i32 %v
}

; The new call should not inherit the range of the first call, since the merged
; index can be 1.
; CHECK-LABEL: define i32 @callsite_index_range(
; CHECK:         %p.idx = select i1 %c, i32 0, i32 1
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 %p.idx)
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %p, align 4
define i32 @callsite_index_range(i1 %c) {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 range(i32 0, 1) 0)
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
}

; A select with an undef arm whose pointer is also compared should stay
; unchanged, even with a valid handle.
; CHECK-LABEL: define noundef i1 @guarded_undef_outside_use(
; CHECK:         %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) undef
; CHECK-NEXT:    br i1 %c, label %read, label %observe
; CHECK:       observe:
; CHECK-NEXT:    %eq = icmp eq ptr addrspace(11) %p, %p0
; CHECK-NEXT:    %result = and i1 %eq, false
; CHECK-NEXT:    ret i1 %result
define noundef i1 @guarded_undef_outside_use(i1 noundef %c) {
entry:
  %slot = alloca ptr addrspace(11)
  %u = load ptr addrspace(11), ptr %slot
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %u
  br i1 %c, label %read, label %observe
read:
  %v = load i32, ptr addrspace(11) %p, align 4
  %nonzero = icmp ne i32 %v, 0
  %defined = freeze i1 %nonzero
  ret i1 %defined
observe:
  %eq = icmp eq ptr addrspace(11) %p, %p0
  %result = and i1 %eq, false
  ret i1 %result
}

; A phi with an undef input whose pointer is compared should stay unchanged,
; since the handle is poison on the undef edge.
; CHECK-LABEL: define noundef i1 @undef_with_poison_handle(
; CHECK:       merge:
; CHECK-NEXT:    %p = phi ptr addrspace(11) [ %p0, %make_ptr ], [ undef, %entry ]
; CHECK-NEXT:    br i1 %c, label %read, label %observe
; CHECK:       observe:
; CHECK-NEXT:    %eq = icmp eq ptr addrspace(11) %p, null
; CHECK-NEXT:    %result = and i1 %eq, false
; CHECK-NEXT:    ret i1 %result
define noundef i1 @undef_with_poison_handle(i1 noundef %c) {
entry:
  %slot = alloca ptr addrspace(11)
  %u = load ptr addrspace(11), ptr %slot
  %buf = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %h = select i1 %c, target("spirv.VulkanBuffer", [0 x i32], 12, 1) %buf, target("spirv.VulkanBuffer", [0 x i32], 12, 1) poison
  br i1 %c, label %make_ptr, label %merge
make_ptr:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %make_ptr ], [ %u, %entry ]
  br i1 %c, label %read, label %observe
read:
  %v = load i32, ptr addrspace(11) %p, align 4
  %nonzero = icmp ne i32 %v, 0
  %defined = freeze i1 %nonzero
  ret i1 %defined
observe:
  %eq = icmp eq ptr addrspace(11) %p, null
  %result = and i1 %eq, false
  ret i1 %result
}

; A pointer with an undef input that is also the stored value should stay
; unchanged, since the store writes the pointer itself.
; CHECK-LABEL: define void @undef_pointer_stored_as_value(
; CHECK:         %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) undef
; CHECK-NEXT:    store ptr addrspace(11) %p, ptr addrspace(11) %p, align 8
define void @undef_pointer_stored_as_value(i1 %c) {
entry:
  %slot = alloca ptr addrspace(11)
  %u = load ptr addrspace(11), ptr %slot
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %u
  store ptr addrspace(11) %p, ptr addrspace(11) %p, align 8
  ret void
}

; A poison input stays poison, and unlike undef it allows other uses of the
; pointer.
; CHECK-LABEL: define i1 @poison_input_compared(
; CHECK:         %p.idx = select i1 %c, i32 1, i32 poison
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 %p.idx)
; CHECK-NEXT:    store i32 0, ptr addrspace(11) %p, align 4
; CHECK-NEXT:    %eq = icmp eq ptr addrspace(11) %p, null
define i1 @poison_input_compared(i1 %c) {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) poison
  store i32 0, ptr addrspace(11) %p, align 4
  %eq = icmp eq ptr addrspace(11) %p, null
  ret i1 %eq
}

; An undef input is allowed when the pointer reaches only accesses through GEPs.
; CHECK-LABEL: define void @undef_input_gep_store(
; CHECK:         %p.idx = select i1 %c, i32 1, i32 0
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %h, i32 %p.idx)
; CHECK-NEXT:    %m = getelementptr inbounds nuw i8, ptr addrspace(11) %p, i64 4
; CHECK-NEXT:    store i32 %v, ptr addrspace(11) %m, align 4
define void @undef_input_gep_store(i1 %c, i32 %v) {
entry:
  %slot = alloca ptr addrspace(11)
  %u = load ptr addrspace(11), ptr %slot
  %h = call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t.i32(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) %h, i32 1)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %u
  %m = getelementptr inbounds nuw i8, ptr addrspace(11) %p, i64 4
  store i32 %v, ptr addrspace(11) %m, align 4
  ret void
}

; The handle for the chosen descriptor dominates both accesses and their merge.
; CHECK-LABEL: define i32 @dominating_non_entry_handle(
; CHECK:       merge:
; CHECK-NEXT:    %p.idx = phi i32 [ 0, %a ], [ 1, %b ]
; CHECK-NEXT:    %p = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 %p.idx)
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %p, align 4
define i32 @dominating_non_entry_handle(i1 %enabled, i1 %c, i32 %slot) {
entry:
  br i1 %enabled, label %lookup, label %exit
lookup:
  %h = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 4, i32 %slot, ptr @.str)
  br i1 %c, label %a, label %b
a:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 0)
  br label %merge
b:
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %a ], [ %p1, %b ]
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
exit:
  ret i32 0
}

; Calls with different index types should stay unchanged.
; CHECK-LABEL: define i32 @mixed_index_types(
; CHECK:         %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %p, align 4
define i32 @mixed_index_types(i1 %c, i32 %i, i64 %j) {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 %i)
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i64(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i64 %j)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1
  %v = load i32, ptr addrspace(11) %p, align 4
  ret i32 %v
}

; A debug record right after the select should follow the new getpointer call
; that replaces the select.
; CHECK-LABEL: define i32 @debug_record_after_select(
; CHECK:         %q.idx = select i1 %c, i32 0, i32 1
; CHECK-NEXT:    %[[Q:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.
; CHECK-NEXT:      #dbg_value(ptr addrspace(11) %[[Q]], !{{[0-9]+}}, !DIExpression(),
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %[[Q]], align 4
define i32 @debug_record_after_select(i1 %c) !dbg !11 {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %q0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0), !dbg !13
  %q1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1), !dbg !13
  %q = select i1 %c, ptr addrspace(11) %q0, ptr addrspace(11) %q1, !dbg !13
    #dbg_value(ptr addrspace(11) %q, !12, !DIExpression(), !13)
  %v = load i32, ptr addrspace(11) %q, align 4, !dbg !14
  ret i32 %v, !dbg !14
}

; A select used only by another select and a debug record should not get a
; getpointer call of its own, since debug records must not change the code.
; Its debug record loses the location instead.
; CHECK-LABEL: define i32 @debug_only_inner_select(
; CHECK-NOT:     %s1 = call
; CHECK-DAG:       #dbg_value(ptr addrspace(11) poison, !{{[0-9]+}}, !DIExpression(),
; CHECK-DAG:     %s2 = call ptr addrspace(11) @llvm.spv.resource.getpointer.
define i32 @debug_only_inner_select(i1 %c, i1 %d) !dbg !15 {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0), !dbg !17
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1), !dbg !17
  %p2 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 2), !dbg !17
  %s1 = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1, !dbg !17
    #dbg_value(ptr addrspace(11) %s1, !16, !DIExpression(), !17)
  %s2 = select i1 %d, ptr addrspace(11) %s1, ptr addrspace(11) %p2, !dbg !17
  %v = load i32, ptr addrspace(11) %s2, align 4, !dbg !17
  ret i32 %v, !dbg !17
}

; The load and the debug record both refer to the replacement pointer.
; CHECK-LABEL: define i32 @debug_pointer_select(
; CHECK:         %p.idx = select i1 %c, i32 0, i32 1
; CHECK-NEXT:    %[[P:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.{{.*}}, !dbg ![[LOC:[0-9]+]]
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %[[P]], align 4
; CHECK-NEXT:      #dbg_value(ptr addrspace(11) %[[P]], ![[VAR:[0-9]+]], !DIExpression(),
; CHECK-NEXT:    ret i32 %v
; CHECK-DAG: ![[VAR]] = !DILocalVariable(name: "p",
; CHECK-DAG: ![[LOC]] = !DILocation(line: 2, column: 3,
define i32 @debug_pointer_select(i1 %c) !dbg !2 {
entry:
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0), !dbg !9
  %p1 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1), !dbg !9
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) %p1, !dbg !9
  %v = load i32, ptr addrspace(11) %p, align 4, !dbg !10
    #dbg_value(ptr addrspace(11) %p, !8, !DIExpression(), !10)
  ret i32 %v, !dbg !10
}

declare target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24), i32)
declare target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ss_12_1t.i32(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 1), i32)
declare target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i64(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i64)
declare target("spirv.VulkanBuffer", %struct.S, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_struct.Ss_2_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(12) @llvm.spv.resource.getpointer.p12.tspirv.VulkanBuffer_s_struct.Ss_2_0t.i32(target("spirv.VulkanBuffer", %struct.S, 2, 0), i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "resource-pointer-phi.c", directory: "/")
!2 = distinct !DISubprogram(name: "debug_pointer_select", scope: !1, file: !1, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!3 = !DISubroutineType(types: !5)
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 32)
!8 = !DILocalVariable(name: "p", scope: !2, file: !1, line: 2, type: !7)
!9 = !DILocation(line: 2, column: 3, scope: !2)
!10 = !DILocation(line: 3, column: 3, scope: !2)
!11 = distinct !DISubprogram(name: "debug_record_after_select", scope: !1, file: !1, line: 10, type: !3, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DILocalVariable(name: "q", scope: !11, file: !1, line: 11, type: !7)
!13 = !DILocation(line: 11, column: 3, scope: !11)
!14 = !DILocation(line: 12, column: 3, scope: !11)
!15 = distinct !DISubprogram(name: "debug_only_inner_select", scope: !1, file: !1, line: 20, type: !3, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!16 = !DILocalVariable(name: "s1", scope: !15, file: !1, line: 21, type: !7)
!17 = !DILocation(line: 21, column: 3, scope: !15)
