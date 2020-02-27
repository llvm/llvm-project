; RUN: opt < %s -instcombine -instcombine-infinite-loop-threshold=2 -S | FileCheck %s

%struct.__va_list = type { i8*, i8*, i8*, i32, i32 }

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
declare void @llvm.va_copy(i8*, i8*)

define i32 @func(i8* nocapture readnone %fmt, ...) {
; CHECK-LABEL: @func(
; CHECK: entry:
; CHECK-NEXT: ret i32 0
entry:
  %va0 = alloca %struct.__va_list, align 8
  %va1 = alloca %struct.__va_list, align 8
  %0 = bitcast %struct.__va_list* %va0 to i8*
  %1 = bitcast %struct.__va_list* %va1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %0)
  call void @llvm.va_start(i8* %0)
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %1)
  call void @llvm.va_copy(i8* %1, i8* %0)
  call void @llvm.va_end(i8* %1)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %1)
  call void @llvm.va_end(i8* %0)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %0)
  ret i32 0
}

