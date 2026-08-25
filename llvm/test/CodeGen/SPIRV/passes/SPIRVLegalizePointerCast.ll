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

; Byte-addressable buffer tests model HLSL ByteAddressBuffer layout ([0 x i8] handles,
; typed load/store via getpointer byte offset). emit-intrinsics adds ptrcast; legalize-pointer-cast removes it.

define void @byteBufferStore() {
; CHECK-LABEL: define void @byteBufferStore(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: store i8 42, ptr addrspace(11)
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
; CHECK: store i8 0, ptr addrspace(11)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store i32 42, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @byteBufferLoad() {
; CHECK-LABEL: define void @byteBufferLoad(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load i8, ptr addrspace(11)
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
; CHECK: load i8, ptr addrspace(11)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load i32, ptr addrspace(11) %ptr, align 4
  ret void
}

@slot = internal global target("spirv.VulkanBuffer", [0 x i8], 12, 0) poison, align 8

define void @byteBufferStoreViaLoadedHandle() {
; CHECK-LABEL: define void @byteBufferStoreViaLoadedHandle(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load target("spirv.VulkanBuffer", [0 x i8], 12, 0), ptr @slot
; CHECK: store i8 42, ptr addrspace(11)
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
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
; CHECK: load i8, ptr addrspace(11)
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  store target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, ptr @slot, align 8
  %loaded = load target("spirv.VulkanBuffer", [0 x i8], 12, 0), ptr @slot, align 8
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %loaded, i32 0)
  %val = load i32, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @byteBufferStore4() {
; CHECK-LABEL: define void @byteBufferStore4(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: store i8
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 4)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr addrspace(11) %ptr, align 16
  ret void
}

define void @byteBufferLoad4() {
; CHECK-LABEL: define void @byteBufferLoad4(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load i8, ptr addrspace(11)
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 4)
; CHECK: call {{.*}}@llvm.spv.insertelt
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load <4 x i32>, ptr addrspace(11) %ptr, align 16
  ret void
}

@outI8 = addrspace(10) global i8 zeroinitializer
@outI32 = addrspace(10) global i32 zeroinitializer
@outF = addrspace(10) global float zeroinitializer

; Single-byte access uses RetagDirect (retag + typed access), not byte-wise
; decomposition.

define void @byteBufferStoreI8() {
; CHECK-LABEL: define void @byteBufferStoreI8(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: store i8 42, ptr addrspace(11)
; CHECK-NOT: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store i8 42, ptr addrspace(11) %ptr, align 1
  ret void
}

define void @byteBufferLoadI8() {
; CHECK-LABEL: define void @byteBufferLoadI8(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load i8, ptr addrspace(11)
; CHECK-NOT: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
; CHECK: store i8 {{.*}}, ptr addrspace(10) @outI8
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load i8, ptr addrspace(11) %ptr, align 1
  store i8 %val, ptr addrspace(10) @outI8, align 1
  ret void
}

; Non-constant getpointer offset forces dynamic add in gepByteOffset.

define void @byteBufferStoreDynamicOffset(i32 %byteOff) {
; CHECK-LABEL: define void @byteBufferStoreDynamicOffset(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: add i32 {{.*}}%byteOff, 1
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 {{%}}
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 %byteOff)
  store i32 42, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @byteBufferLoadDynamicOffset(i32 %byteOff) {
; CHECK-LABEL: define void @byteBufferLoadDynamicOffset(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: add i32 {{.*}}%byteOff, 1
; CHECK: load i8, ptr addrspace(11)
; CHECK: store i32 {{.*}}, ptr addrspace(10) @outI32
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 %byteOff)
  %val = load i32, ptr addrspace(11) %ptr, align 4
  store i32 %val, ptr addrspace(10) @outI32, align 4
  ret void
}

; Float access exercises bitcast in byte-wise combine/decompose paths.

define void @byteBufferStoreFloat() {
; CHECK-LABEL: define void @byteBufferStoreFloat(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: store i8
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 3)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store float 1.000000e+00, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @byteBufferLoadFloat() {
; CHECK-LABEL: define void @byteBufferLoadFloat(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load i8, ptr addrspace(11)
; CHECK: bitcast i32 {{.*}} to float
; CHECK: store float {{.*}}, ptr addrspace(10) @outF
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load float, ptr addrspace(11) %ptr, align 4
  store float %val, ptr addrspace(10) @outF, align 4
  ret void
}

@outI16 = addrspace(10) global i16 zeroinitializer, align 2
@outV2I16 = addrspace(10) global <2 x i16> zeroinitializer, align 4

; i16 access uses byte-wise decomposition (2 bytes).

define void @byteBufferStoreI16() {
; CHECK-LABEL: define void @byteBufferStoreI16(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: store i8 42, ptr addrspace(11)
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
; CHECK: store i8 0, ptr addrspace(11)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store i16 42, ptr addrspace(11) %ptr, align 2
  ret void
}

define void @byteBufferLoadI16() {
; CHECK-LABEL: define void @byteBufferLoadI16(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load i8, ptr addrspace(11)
; CHECK: zext i8 {{.*}} to i16
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
; CHECK: store i16 {{.*}}, ptr addrspace(10) @outI16
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load i16, ptr addrspace(11) %ptr, align 2
  store i16 %val, ptr addrspace(10) @outI16, align 2
  ret void
}

; <2 x i16> vector access decomposes per element (4 bytes total).

define void @byteBufferStoreV2I16() {
; CHECK-LABEL: define void @byteBufferStoreV2I16(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: trunc i16 {{.*}} to i8
; CHECK: store i8 {{.*}}, ptr addrspace(11)
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 3)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store <2 x i16> <i16 1, i16 2>, ptr addrspace(11) %ptr, align 4
  ret void
}

define void @byteBufferLoadV2I16() {
; CHECK-LABEL: define void @byteBufferLoadV2I16(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: load i8, ptr addrspace(11)
; CHECK: zext i8 {{.*}} to i16
; CHECK: call {{.*}}@llvm.spv.insertelt
; CHECK: call void @llvm.spv.store.v2i16.p10
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %val = load <2 x i16>, ptr addrspace(11) %ptr, align 4
  store <2 x i16> %val, ptr addrspace(10) @outV2I16, align 4
  ret void
}

; Same base pointer, i32 then i16 typed accesses (distinct ptrcasts).

define void @byteBufferMixedI32ThenI16Store() {
; CHECK-LABEL: define void @byteBufferMixedI32ThenI16Store(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: store i8 1, ptr addrspace(11)
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 3)
; CHECK: store i8 2, ptr addrspace(11)
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  store i32 1, ptr addrspace(11) %ptr, align 4
  store i16 2, ptr addrspace(11) %ptr, align 2
  ret void
}

define void @byteBufferMixedLoadI32ThenI16() {
; CHECK-LABEL: define void @byteBufferMixedLoadI32ThenI16(
; CHECK-NOT: call {{.*}}@llvm.spv.ptrcast
; CHECK: zext i8 {{.*}} to i32
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}} i32 1)
; CHECK: zext i8 {{.*}} to i16
; CHECK: shl i16
entry:
  %handle = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %handle, i32 0)
  %i32val = load i32, ptr addrspace(11) %ptr, align 4
  %i16val = load i16, ptr addrspace(11) %ptr, align 2
  store i32 %i32val, ptr addrspace(10) @outI32, align 4
  store i16 %i16val, ptr addrspace(10) @outI16, align 2
  ret void
}
