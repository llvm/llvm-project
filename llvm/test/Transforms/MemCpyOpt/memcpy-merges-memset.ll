; RUN: opt -opaque-pointers -memcpyopt -S < %s -verify-memoryssa | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"

%struct.deliver_proc_ctx_t = type { i32, [1024 x i32] }
define dso_local void @test(ptr noundef writeonly %ctx) local_unnamed_addr {
; CHECK-LABEL: @test(
; CHECK-LABEL: entry
; CHECK-NEXT:    %.compoundliteral = alloca %struct.deliver_proc_ctx_t, align 4
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4100) %ctx, i8 0, i64 4100, i1 false)
; CHECK-NEXT:    ret void
;
entry:
  %.compoundliteral = alloca %struct.deliver_proc_ctx_t, align 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4100) %.compoundliteral, i8 0, i64 4100, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %ctx, ptr nonnull align 4 %.compoundliteral, i64 4100, i1 true)
  ret void
}

declare void @llvm.memset.p0.i64(ptr writeonly, i8, i64, i1 immarg)

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly, ptr noalias readonly, i64, i1 immarg)