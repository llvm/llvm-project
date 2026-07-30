; spirv-legalize-pointer-cast consumes spv.ptrcast intrinsics produced by
; spirv-emit-intrinsics, so we chain both passes and check the ptrcast is
; rewritten into a sequence of typed loads + gep/extractelt.
;
; RUN: opt -S -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s

@M = internal addrspace(10) global [4 x <2 x float>] zeroinitializer, align 4
@OUT = internal addrspace(10) global float zeroinitializer, align 4
@Arr = internal addrspace(10) global [8 x float] zeroinitializer, align 16
@OUTV = internal addrspace(10) global <4 x float> zeroinitializer, align 4

; Loading a <5 x float> through a [4 x <2 x float>] forces emit-intrinsics to
; insert spv.ptrcast; legalize-pointer-cast lowers it to typed <2 x float>
; loads stitched together with extractelt. After the pass, no spv.ptrcast call
; should remain.

define spir_func void @main() #0 {
; CHECK-LABEL: define spir_func void @main(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: call ptr addrspace(10) {{.*}}@llvm.spv.gep.p10.p10(i1 false, ptr addrspace(10) @M, i32 0, i32 0)
; CHECK: load <2 x float>, ptr addrspace(10)
; CHECK: call float @llvm.spv.extractelt.f32.v2f32.i32(<2 x float>
entry:
  %v = load <5 x float>, ptr addrspace(10) @M, align 4
  %x = extractelement <5 x float> %v, i32 4
  store float %x, ptr addrspace(10) @OUT, align 4
  ret void
}

; Loading a <4 x float> from an [8 x float] with a base alignment of 16 must
; not strengthen or discard alignment on the split per-element loads: the
; alignment of each load is the common alignment of the base align and its
; byte offset (16, 4, 8, 4 for offsets 0, 4, 8, 12).

define spir_func void @loadAlign() #0 {
; CHECK-LABEL: define spir_func void @loadAlign(
; CHECK: load float, ptr addrspace(10) %{{.*}}, align 16
; CHECK: load float, ptr addrspace(10) %{{.*}}, align 4
; CHECK: load float, ptr addrspace(10) %{{.*}}, align 8
; CHECK: load float, ptr addrspace(10) %{{.*}}, align 4
entry:
  %v = load <4 x float>, ptr addrspace(10) @Arr, align 16
  store <4 x float> %v, ptr addrspace(10) @OUTV, align 4
  ret void
}

; Storing a <4 x float> into an [8 x float] with a base alignment of 16 must
; apply the same commonAlignment rule to the split per-element stores.

define spir_func void @storeAlign() #0 {
; CHECK-LABEL: define spir_func void @storeAlign(
; CHECK: store float %{{.*}}, ptr addrspace(10) %{{.*}}, align 16
; CHECK: store float %{{.*}}, ptr addrspace(10) %{{.*}}, align 4
; CHECK: store float %{{.*}}, ptr addrspace(10) %{{.*}}, align 8
; CHECK: store float %{{.*}}, ptr addrspace(10) %{{.*}}, align 4
entry:
  %v = load <4 x float>, ptr addrspace(10) @OUTV, align 4
  store <4 x float> %v, ptr addrspace(10) @Arr, align 16
  ret void
}

@WIDEN = external addrspace(12) global <{ <1 x float>, target("spirv.Padding", 12), <1 x float> }>, align 4

define spir_func void @widen() #0 {
; CHECK-LABEL: define spir_func void @widen(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: call ptr addrspace(12) {{.*}}@llvm.spv.gep.p12.p12(i1 false, ptr addrspace(12) @WIDEN, i32 0, i32 0)
; CHECK: load <1 x float>, ptr addrspace(12)
; CHECK: call float @llvm.spv.bitcast.f32.v1f32(<1 x float>
; CHECK: call <4 x float> @llvm.spv.insertelt.v4f32.v4f32.f32.i32(<4 x float> poison, float
entry:
  %v = load <4 x float>, ptr addrspace(12) @WIDEN, align 4
  %x = extractelement <4 x float> %v, i32 0
  store float %x, ptr addrspace(10) @OUT, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

declare target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0), i32)

; Byte-addressable buffer tests model Clang's SPIR-V resource layout for HLSL
; ByteAddressBuffer/RWByteAddressBuffer: the handle carries [0 x i8] as the
; OpTypeRuntimeArray element type for storage-buffer blocks. That i8 names
; byte-addressable layout in LLVM IR; it is not an HLSL surface type. Typed
; HLSL accesses (e.g. Store(offset, uint)) lower to i32 load/store at a byte
; offset in getpointer's index operand. emit-intrinsics tags getpointer as i8
; from the layout, then inserts i8->T spv_ptrcast for the typed access;
; legalize-pointer-cast removes the ptrcast and keeps the typed access.

define void @byteBufferStore() {
; CHECK-LABEL: define void @byteBufferStore(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: store i32 42, ptr addrspace(11)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store i32 42, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @byteBufferLoad() {
; CHECK-LABEL: define void @byteBufferLoad(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load i32, ptr addrspace(11)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load i32, ptr addrspace(11) %ptr, align 4
  ret void
}

@slot = internal global target("spirv.VulkanBuffer", [0 x i8], 12, 0) poison, align 8

; Regression for issue #192523: handle flows through memory before getpointer
; (as in local RWByteAddressBuffer arrays), then typed store hits i8->i32 ptrcast.

define void @byteBufferStoreViaLoadedHandle() {
; CHECK-LABEL: define void @byteBufferStoreViaLoadedHandle(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load target("spirv.VulkanBuffer", [0 x i8], 12, 0), ptr @slot
; CHECK: store i32 42, ptr addrspace(11)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  store target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, ptr @slot, align 8
  %loaded = load target("spirv.VulkanBuffer", [0 x i8], 12, 0), ptr @slot, align 8
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %loaded, i32 0)
  store i32 42, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @byteBufferLoadViaLoadedHandle() {
; CHECK-LABEL: define void @byteBufferLoadViaLoadedHandle(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load target("spirv.VulkanBuffer", [0 x i8], 12, 0), ptr @slot
; CHECK: load i32, ptr addrspace(11)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  store target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, ptr @slot, align 8
  %loaded = load target("spirv.VulkanBuffer", [0 x i8], 12, 0), ptr @slot, align 8
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %loaded, i32 0)
  %val = load i32, ptr addrspace(11) %ptr, align 4
  ret void
}
