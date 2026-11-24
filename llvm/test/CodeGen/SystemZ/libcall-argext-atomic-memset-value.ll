; RUN: llc < %s -mtriple=s390x-linux-gnu -argext-abi-check | FileCheck %s
; REQUIRES: asserts
;
; The i8 value argument passed to the __llvm_memset_element_unordered_atomic_N
; libcalls must carry an extension attribute, as required by the SystemZ C ABI.

define void @test_atomic_memset_value(ptr %Dst, i64 %Len) {
; CHECK-LABEL: test_atomic_memset_value:
; CHECK: lghi %r3, 16
; CHECK: brasl %r14, __llvm_memset_element_unordered_atomic_1@PLT
  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 1 %Dst, i8 16, i64 %Len, i32 1)
  ret void
}

declare void @llvm.memset.element.unordered.atomic.p0.i64(ptr nocapture writeonly, i8, i64, i32 immarg)
