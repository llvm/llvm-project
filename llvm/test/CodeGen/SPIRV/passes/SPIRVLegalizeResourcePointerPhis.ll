; RUN: opt -S -verify-each -passes='function(spirv-legalize-resource-pointer-phis)' -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s

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

; The loop-carried index phi's entry value is bypassed on the first iteration.
; CHECK-LABEL: define i32 @loop_phi(
; CHECK:       header:
; CHECK-NEXT:    %q.idx = phi i32 [ {{[^,]+}}, %entry ], [ %m.idx, %body ]
; CHECK-NEXT:    %i = phi i32
; CHECK:       merge:
; CHECK-NEXT:    %m.idx = phi i32 [ %q.idx, %header ], [ 0, %a ]
; CHECK-NEXT:    %m = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_24t.i32(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) %h, i32 %m.idx)
; CHECK:       body:
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %m, align 4
define i32 @loop_phi(i32 %n) {
entry:
  %h = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_24t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  br label %header
header:
  %q = phi ptr addrspace(11) [ undef, %entry ], [ %m, %body ]
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

; The backedge brings a pointer to the previous iteration's descriptor.
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

; A select with only undefined inputs cannot supply a resource handle.
; CHECK-LABEL: define void @no_getpointer(
; CHECK:         %p = select i1 %c, ptr addrspace(11) undef, ptr addrspace(11) poison
; CHECK:         store i32 0, ptr addrspace(11) %p, align 4
define void @no_getpointer(i1 %c, i1 %access) {
entry:
  %p = select i1 %c, ptr addrspace(11) undef, ptr addrspace(11) poison
  br i1 %access, label %write, label %exit
write:
  store i32 0, ptr addrspace(11) %p, align 4
  br label %exit
exit:
  ret void
}

; Different constant-buffer members must keep constant member indices.
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

; The merged index can be 1, outside the first call site's range.
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
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  %p = select i1 %c, ptr addrspace(11) %p0, ptr addrspace(11) undef
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

; The undef path's comparison stays defined even when the handle is poison.
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
  %buf = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %h = select i1 %c, target("spirv.VulkanBuffer", [0 x i32], 12, 1) %buf, target("spirv.VulkanBuffer", [0 x i32], 12, 1) poison
  br i1 %c, label %make_ptr, label %merge
make_ptr:
  %p0 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t.i32(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 1)
  br label %merge
merge:
  %p = phi ptr addrspace(11) [ %p0, %make_ptr ], [ undef, %entry ]
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

; Leave mixed-width indices unchanged because they require a common index type.
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

; The load and the debug record both refer to the replacement pointer.
; CHECK-LABEL: define i32 @debug_pointer_select(
; CHECK:         %p.idx = select i1 %c, i32 0, i32 1
; CHECK-NEXT:    %[[P:.*]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.
; CHECK-NEXT:    %v = load i32, ptr addrspace(11) %[[P]], align 4
; CHECK-NEXT:      #dbg_value(ptr addrspace(11) %[[P]], ![[VAR:[0-9]+]], !DIExpression(),
; CHECK-NEXT:    ret i32 %v
; CHECK: ![[VAR]] = !DILocalVariable(name: "p",
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
