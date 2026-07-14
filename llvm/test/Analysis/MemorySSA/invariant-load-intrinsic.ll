; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s

declare void @clobber(ptr)

define <4 x i32> @masked_load_invariant(ptr %p, <4 x i1> %mask, <4 x i32> %passthru) {
; CHECK-LABEL: define <4 x i32> @masked_load_invariant(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobber(ptr %p)
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 %p, <4 x i1> %mask, <4 x i32> %passthru), !invariant.load !0
  call void @clobber(ptr %p)
  %v = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 %p, <4 x i1> %mask, <4 x i32> %passthru), !invariant.load !0
  ret <4 x i32> %v
}

define <4 x i32> @masked_load_without_metadata(ptr %p, <4 x i1> %mask, <4 x i32> %passthru) {
; CHECK-LABEL: define <4 x i32> @masked_load_without_metadata(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobber(ptr %p)
; CHECK: MemoryUse(1)
; CHECK-NEXT: %v = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 %p, <4 x i1> %mask, <4 x i32> %passthru)
  call void @clobber(ptr %p)
  %v = call <4 x i32> @llvm.masked.load.v4i32.p0(ptr align 4 %p, <4 x i1> %mask, <4 x i32> %passthru)
  ret <4 x i32> %v
}

define <4 x i32> @masked_gather_invariant(ptr %p, <4 x ptr> %ptrs, <4 x i1> %mask, <4 x i32> %passthru) {
; CHECK-LABEL: define <4 x i32> @masked_gather_invariant(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobber(ptr %p)
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> align 4 %ptrs, <4 x i1> %mask, <4 x i32> %passthru), !invariant.load !0
  call void @clobber(ptr %p)
  %v = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> align 4 %ptrs, <4 x i1> %mask, <4 x i32> %passthru), !invariant.load !0
  ret <4 x i32> %v
}

define <4 x i32> @masked_expandload_invariant(ptr %p, <4 x i1> %mask, <4 x i32> %passthru) {
; CHECK-LABEL: define <4 x i32> @masked_expandload_invariant(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobber(ptr %p)
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v = call <4 x i32> @llvm.masked.expandload.v4i32.p0(ptr %p, <4 x i1> %mask, <4 x i32> %passthru), !invariant.load !0
  call void @clobber(ptr %p)
  %v = call <4 x i32> @llvm.masked.expandload.v4i32.p0(ptr %p, <4 x i1> %mask, <4 x i32> %passthru), !invariant.load !0
  ret <4 x i32> %v
}

define <4 x i32> @vp_gather_invariant(ptr %p, <4 x ptr> %ptrs, <4 x i1> %mask, i32 %vl) {
; CHECK-LABEL: define <4 x i32> @vp_gather_invariant(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: call void @clobber(ptr %p)
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v = call <4 x i32> @llvm.vp.gather.v4i32.v4p0(<4 x ptr> %ptrs, <4 x i1> %mask, i32 %vl), !invariant.load !0
  call void @clobber(ptr %p)
  %v = call <4 x i32> @llvm.vp.gather.v4i32.v4p0(<4 x ptr> %ptrs, <4 x i1> %mask, i32 %vl), !invariant.load !0
  ret <4 x i32> %v
}

!0 = !{}
