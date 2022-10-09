; RUN: opt < %s -dse -S -enable-dse-partial-overwrite-tracking | FileCheck %s
; PR28588

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @_UPT_destroy(ptr nocapture %ptr) local_unnamed_addr #0 {
entry:
  %edi = getelementptr inbounds i8, ptr %ptr, i64 8

; CHECK-NOT: tail call void @llvm.memset.p0.i64(ptr align 8 %edi, i8 0, i64 176, i1 false)
; CHECK-NOT: store i32 -1, ptr %addr

  tail call void @llvm.memset.p0.i64(ptr align 8 %edi, i8 0, i64 176, i1 false)
  %format4.i = getelementptr inbounds i8, ptr %ptr, i64 144
  store i32 -1, ptr %format4.i, align 8

; CHECK: tail call void @free
  tail call void @free(ptr nonnull %ptr)
  ret void
}

; Function Attrs: nounwind
declare void @free(ptr nocapture allocptr) local_unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #1

attributes #0 = { nounwind allockind("free")}
attributes #1 = { argmemonly nounwind }
