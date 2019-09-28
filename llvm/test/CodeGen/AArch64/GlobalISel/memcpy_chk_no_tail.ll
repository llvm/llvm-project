; RUN: llc -global-isel -verify-machineinstrs %s -o - | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios13.0.0"

%struct.int_sqrt = type { i32, i32 }

; Function Attrs: nounwind optsize ssp uwtable
; CHECK-LABEL: @usqrt
; CHECK-NOT: b memcpy
; CHECK: bl _memcpy
define void @usqrt(i32 %x, %struct.int_sqrt* %q) local_unnamed_addr #0 {
  %a = alloca i32, align 4
  %bc = bitcast i32* %a to i8*
  %bc2 = bitcast %struct.int_sqrt* %q to i8*
  %obj = tail call i64 @llvm.objectsize.i64.p0i8(i8* %bc2, i1 false, i1 true, i1 false)
  %call = call i8* @__memcpy_chk(i8* %bc2, i8* nonnull %bc, i64 1000, i64 %obj) #4
  ret void
}

; Function Attrs: nofree nounwind optsize
declare i8* @__memcpy_chk(i8*, i8*, i64, i64) local_unnamed_addr #2

; Function Attrs: nounwind readnone speculatable willreturn
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1 immarg, i1 immarg, i1 immarg) #3
attributes #0 = { optsize "disable-tail-calls"="false" "frame-pointer"="all" }
attributes #2 = { nofree nounwind "disable-tail-calls"="false" "frame-pointer"="all" }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { nounwind optsize }

