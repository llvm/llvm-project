; RUN: llc < %s -mtriple=s390x-linux-gnu -argext-abi-check | FileCheck %s
; REQUIRES: asserts
;
; The length argument of the element-wise atomic memcpy/memmove/memset
; intrinsics can be i8, i16 or i32. When lowered to the
; __llvm_mem*_element_unordered_atomic_N libcalls it is passed as a narrow
; integer argument, which must carry an extension attribute per the SystemZ C
; ABI.
;
; Note: the memset functions below also pass an i8 value argument, so they
; depend on that argument being extended as well (see
; libcall-argext-atomic-memset-value.ll).

; --- memcpy ---

define void @test_atomic_memcpy_i8(ptr %Dst, ptr %Src, i8 %Len) {
; CHECK-LABEL: test_atomic_memcpy_i8:
; CHECK: llgcr %r4, %r4
; CHECK: brasl %r14, __llvm_memcpy_element_unordered_atomic_1@PLT
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i8(ptr align 1 %Dst, ptr align 1 %Src, i8 %Len, i32 1)
  ret void
}

define void @test_atomic_memcpy_i16(ptr %Dst, ptr %Src, i16 %Len) {
; CHECK-LABEL: test_atomic_memcpy_i16:
; CHECK: llghr %r4, %r4
; CHECK: brasl %r14, __llvm_memcpy_element_unordered_atomic_1@PLT
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i16(ptr align 1 %Dst, ptr align 1 %Src, i16 %Len, i32 1)
  ret void
}

define void @test_atomic_memcpy_i32(ptr %Dst, ptr %Src, i32 %Len) {
; CHECK-LABEL: test_atomic_memcpy_i32:
; CHECK: llgfr %r4, %r4
; CHECK: brasl %r14, __llvm_memcpy_element_unordered_atomic_1@PLT
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr align 1 %Dst, ptr align 1 %Src, i32 %Len, i32 1)
  ret void
}

; --- memmove ---

define void @test_atomic_memmove_i8(ptr %Dst, ptr %Src, i8 %Len) {
; CHECK-LABEL: test_atomic_memmove_i8:
; CHECK: llgcr %r4, %r4
; CHECK: brasl %r14, __llvm_memmove_element_unordered_atomic_1@PLT
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i8(ptr align 1 %Dst, ptr align 1 %Src, i8 %Len, i32 1)
  ret void
}

define void @test_atomic_memmove_i16(ptr %Dst, ptr %Src, i16 %Len) {
; CHECK-LABEL: test_atomic_memmove_i16:
; CHECK: llghr %r4, %r4
; CHECK: brasl %r14, __llvm_memmove_element_unordered_atomic_1@PLT
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i16(ptr align 1 %Dst, ptr align 1 %Src, i16 %Len, i32 1)
  ret void
}

define void @test_atomic_memmove_i32(ptr %Dst, ptr %Src, i32 %Len) {
; CHECK-LABEL: test_atomic_memmove_i32:
; CHECK: llgfr %r4, %r4
; CHECK: brasl %r14, __llvm_memmove_element_unordered_atomic_1@PLT
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr align 1 %Dst, ptr align 1 %Src, i32 %Len, i32 1)
  ret void
}

; --- memset ---

define void @test_atomic_memset_i8(ptr %Dst, i8 %Val, i8 %Len) {
; CHECK-LABEL: test_atomic_memset_i8:
; CHECK: llgcr %r4, %r4
; CHECK: brasl %r14, __llvm_memset_element_unordered_atomic_1@PLT
  call void @llvm.memset.element.unordered.atomic.p0.i8(ptr align 1 %Dst, i8 %Val, i8 %Len, i32 1)
  ret void
}

define void @test_atomic_memset_i16(ptr %Dst, i8 %Val, i16 %Len) {
; CHECK-LABEL: test_atomic_memset_i16:
; CHECK: llghr %r4, %r4
; CHECK: brasl %r14, __llvm_memset_element_unordered_atomic_1@PLT
  call void @llvm.memset.element.unordered.atomic.p0.i16(ptr align 1 %Dst, i8 %Val, i16 %Len, i32 1)
  ret void
}

define void @test_atomic_memset_i32(ptr %Dst, i8 %Val, i32 %Len) {
; CHECK-LABEL: test_atomic_memset_i32:
; CHECK: llgfr %r4, %r4
; CHECK: brasl %r14, __llvm_memset_element_unordered_atomic_1@PLT
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 1 %Dst, i8 %Val, i32 %Len, i32 1)
  ret void
}

declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i8(ptr nocapture writeonly, ptr nocapture readonly, i8, i32 immarg)
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i16(ptr nocapture writeonly, ptr nocapture readonly, i16, i32 immarg)
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i32 immarg)
declare void @llvm.memmove.element.unordered.atomic.p0.p0.i8(ptr nocapture writeonly, ptr nocapture readonly, i8, i32 immarg)
declare void @llvm.memmove.element.unordered.atomic.p0.p0.i16(ptr nocapture writeonly, ptr nocapture readonly, i16, i32 immarg)
declare void @llvm.memmove.element.unordered.atomic.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i32 immarg)
declare void @llvm.memset.element.unordered.atomic.p0.i8(ptr nocapture writeonly, i8, i8, i32 immarg)
declare void @llvm.memset.element.unordered.atomic.p0.i16(ptr nocapture writeonly, i8, i16, i32 immarg)
declare void @llvm.memset.element.unordered.atomic.p0.i32(ptr nocapture writeonly, i8, i32, i32 immarg)
