; RUN: opt < %s -basic-aa -dse -enable-dse-memoryssa=false -S -enable-dse-partial-overwrite-tracking | FileCheck %s
; PR28588

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @_UPT_destroy(i8* nocapture %ptr) local_unnamed_addr #0 {
entry:
  %edi = getelementptr inbounds i8, i8* %ptr, i64 8

; CHECK-NOT: tail call void @llvm.memset.p0i8.i64(i8* align 8 %edi, i8 0, i64 176, i1 false)
; CHECK-NOT: store i32 -1, i32* %addr

  tail call void @llvm.memset.p0i8.i64(i8* align 8 %edi, i8 0, i64 176, i1 false)
  %format4.i = getelementptr inbounds i8, i8* %ptr, i64 144
  %addr = bitcast i8* %format4.i to i32*
  store i32 -1, i32* %addr, align 8

; CHECK: tail call void @free
  tail call void @free(i8* nonnull %ptr)
  ret void
}

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
