; RUN: llc -march=hexagon -mhvx < %s | FileCheck %s

; Check that Hexagon Vector Combine do not create vmem for adjacent
; loads less than 4.

; Test 1: We have three adjacent loads. By default, we won't produce vmem
; for loads less than 4.
; CHECK-LABEL: @foo1
; CHECK: vmemu(
; CHECK-NOT: vmem(

define <32 x i32> @foo1(ptr %iptr) #0 align 32 {
entry:
  %0 = load <32 x i32>, ptr %iptr, align 1
  %add.ptr = getelementptr inbounds i8, ptr %iptr, i32 128
  %1 = load <32 x i32>, ptr %add.ptr, align 1
  %add.ptr2 = getelementptr inbounds i8, ptr %iptr, i32 256
  %2 = load <32 x i32>, ptr %add.ptr2, align 1
  %3 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %1, <32 x i32> %0)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %3)
  %5 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %2, <32 x i32> %1)
  %6 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %5)
  %7 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %4, <32 x i32> %6)
  %8 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %7)
  ret <32 x i32> %8
}

; Test 2: We have four adjacent loads. Thus, we will produce vmem.
; CHECK-LABEL: @foo2
; CHECK: vmem(
; CHECK-NOT: vmemu

define <32 x i32> @foo2(ptr %iptr) #0 align 32 {
entry:
  %0 = load <32 x i32>, ptr %iptr, align 1
  %add.ptr = getelementptr inbounds i8, ptr %iptr, i32 128
  %1 = load <32 x i32>, ptr %add.ptr, align 1
  %add.ptr2 = getelementptr inbounds i8, ptr %iptr, i32 256
  %2 = load <32 x i32>, ptr %add.ptr2, align 1
  %add.ptr4 = getelementptr inbounds i8, ptr %iptr, i32 512
  %3 = load <32 x i32>, ptr %add.ptr4, align 1
  %4 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %1, <32 x i32> %0)
  %5 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %4)
  %6 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %3, <32 x i32> %2)
  %7 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %6)
  %8 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %5, <32 x i32> %7)
  %9 = tail call <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32> %8)
  ret <32 x i32> %9
}

declare <32 x i32> @llvm.hexagon.V6.vconv.hf.qf32.128B(<64 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1
attributes #0 = {"target-features"="+hvx-length128b,+hvxv75" }
