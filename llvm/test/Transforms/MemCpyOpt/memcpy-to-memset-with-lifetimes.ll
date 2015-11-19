; RUN: opt -basicaa -memcpyopt -instcombine -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo([8 x i64]* noalias nocapture sret dereferenceable(64) %sret) {
entry-block:
  %a = alloca [8 x i64], align 8
  %a.cast = bitcast [8 x i64]* %a to i8*
  call void @llvm.lifetime.start(i64 64, i8* %a.cast)
  call void @llvm.memset.p0i8.i64(i8* %a.cast, i8 0, i64 64, i1 false)
  %sret.cast = bitcast [8 x i64]* %sret to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %sret.cast, i8* %a.cast, i64 64, i1 false)
  call void @llvm.lifetime.end(i64 64, i8* %a.cast)
  ret void

; CHECK-LABEL: @foo(
; CHECK:         %[[sret_cast:[^=]+]] = bitcast [8 x i64]* %sret to i8*
; CHECK-NEXT:    call void @llvm.memset.p0i8.i64(i8* align 8 %[[sret_cast]], i8 0, i64 64
; CHECK-NOT: call void @llvm.memcpy
; CHECK: ret void
}

define void @bar([8 x i64]* noalias nocapture sret dereferenceable(64) %sret, [8 x i64]* noalias nocapture dereferenceable(64) %out) {
entry-block:
  %a = alloca [8 x i64], align 8
  %a.cast = bitcast [8 x i64]* %a to i8*
  call void @llvm.lifetime.start(i64 64, i8* %a.cast)
  call void @llvm.memset.p0i8.i64(i8* align 8 %a.cast, i8 0, i64 64, i1 false)
  %sret.cast = bitcast [8 x i64]* %sret to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %sret.cast, i8* align 8 %a.cast, i64 64, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 8 %a.cast, i8 42, i64 32, i1 false)
  %out.cast = bitcast [8 x i64]* %out to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %out.cast, i8* align 8 %a.cast, i64 64, i1 false)
  call void @llvm.lifetime.end(i64 64, i8* %a.cast)
  ret void

; CHECK-LABEL: @bar(
; CHECK:         %[[a:[^=]+]] = alloca [8 x i64]
; CHECK:         %[[a_cast:[^=]+]] = bitcast [8 x i64]* %[[a]] to i8*
; CHECK:         call void @llvm.memset.p0i8.i64(i8* align 8 %[[a_cast]], i8 0, i64 64
; CHECK:         %[[sret_cast:[^=]+]] = bitcast [8 x i64]* %sret to i8*
; CHECK:         call void @llvm.memset.p0i8.i64(i8* align 8 %[[sret_cast]], i8 0, i64 64
; CHECK:         call void @llvm.memset.p0i8.i64(i8* align 8 %[[a_cast]], i8 42, i64 32
; CHECK:         %[[out_cast:[^=]+]] = bitcast [8 x i64]* %out to i8*
; CHECK:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %[[out_cast]], i8* align 8 %[[a_cast]], i64 64
; CHECK-NOT: call void @llvm.memcpy
; CHECK: ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
