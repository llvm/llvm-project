; RUN: opt -passes=rewrite-statepoints-for-gc -rs4gc-allow-statepoint-with-no-deopt-info=0 -S < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-REQUIRE-DEOPT
; RUN: opt -passes=rewrite-statepoints-for-gc -rs4gc-allow-statepoint-with-no-deopt-info=1 -S < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NO-REQUIRE-DEOPT

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare void @llvm.memcpy.element.unordered.atomic.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i32 immarg)
declare void @llvm.memmove.element.unordered.atomic.p1.p1.i32(ptr addrspace(1), ptr addrspace(1), i32, i32 immarg)

define void @test_memcpy_no_deopt(ptr addrspace(1) %src, i64 %src_offset, ptr addrspace(1) %dest, i64 %dest_offset, i32 %len) gc "statepoint-example" {
; CHECK-LABEL: @test_memcpy_no_deopt
; CHECK-REQUIRE-DEOPT-NOT: @llvm.experimental.gc.statepoint
; CHECK-NO-REQUIRE-DEOPT: @llvm.experimental.gc.statepoint
entry:
  %src_derived = getelementptr inbounds i8, ptr addrspace(1) %src, i64 %src_offset
  %dest_derived = getelementptr inbounds i8, ptr addrspace(1) %dest, i64 %dest_offset
  call void @llvm.memcpy.element.unordered.atomic.p1.p1.i32(ptr addrspace(1) align 16 %src_derived, ptr addrspace(1) align 16 %dest_derived, i32 %len, i32 1)
  ret void
}

define void @test_memmove_no_deopt(ptr addrspace(1) %src, i64 %src_offset, ptr addrspace(1) %dest, i64 %dest_offset, i32 %len) gc "statepoint-example" {
; CHECK-LABEL: @test_memmove_no_deopt
; CHECK-REQUIRE-DEOPT-NOT: @llvm.experimental.gc.statepoint
; CHECK-NO-REQUIRE-DEOPT: @llvm.experimental.gc.statepoint
entry:
  %src_derived = getelementptr inbounds i8, ptr addrspace(1) %src, i64 %src_offset
  %dest_derived = getelementptr inbounds i8, ptr addrspace(1) %dest, i64 %dest_offset
  call void @llvm.memmove.element.unordered.atomic.p1.p1.i32(ptr addrspace(1) align 16 %src_derived, ptr addrspace(1) align 16 %dest_derived, i32 %len, i32 1)
  ret void
}

define void @test_memcpy_with_deopt(ptr addrspace(1) %src, i64 %src_offset, ptr addrspace(1) %dest, i64 %dest_offset, i32 %len) gc "statepoint-example" {
; CHECK-LABEL: @test_memcpy_with_deopt
; CHECK-REQUIRE-DEOPT: @llvm.experimental.gc.statepoint
; CHECK-NO-REQUIRE-DEOPT: @llvm.experimental.gc.statepoint
entry:
  %src_derived = getelementptr inbounds i8, ptr addrspace(1) %src, i64 %src_offset
  %dest_derived = getelementptr inbounds i8, ptr addrspace(1) %dest, i64 %dest_offset
  call void @llvm.memcpy.element.unordered.atomic.p1.p1.i32(ptr addrspace(1) align 16 %src_derived, ptr addrspace(1) align 16 %dest_derived, i32 %len, i32 1) [ "deopt"(i32 0) ]
  ret void
}

define void @test_memmove_with_deopt(ptr addrspace(1) %src, i64 %src_offset, ptr addrspace(1) %dest, i64 %dest_offset, i32 %len) gc "statepoint-example" {
; CHECK-LABEL: @test_memmove_with_deopt
; CHECK-REQUIRE-DEOPT: @llvm.experimental.gc.statepoint
; CHECK-NO-REQUIRE-DEOPT: @llvm.experimental.gc.statepoint
entry:
  %src_derived = getelementptr inbounds i8, ptr addrspace(1) %src, i64 %src_offset
  %dest_derived = getelementptr inbounds i8, ptr addrspace(1) %dest, i64 %dest_offset
  call void @llvm.memmove.element.unordered.atomic.p1.p1.i32(ptr addrspace(1) align 16 %src_derived, ptr addrspace(1) align 16 %dest_derived, i32 %len, i32 1) [ "deopt"(i32 0) ]
  ret void
}
