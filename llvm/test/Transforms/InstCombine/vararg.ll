; RUN: opt < %s -passes=instcombine -S | FileCheck %s

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)
declare void @llvm.va_copy(ptr, ptr)

define i32 @func(ptr nocapture readnone %fmt, ...) {
; CHECK-LABEL: @func(
; CHECK: entry:
; CHECK-NEXT: ret i32 0
entry:
  %va0 = alloca %struct.__va_list, align 8
  %va1 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr %va0)
  call void @llvm.va_start(ptr %va0)
  call void @llvm.lifetime.start.p0(i64 32, ptr %va1)
  call void @llvm.va_copy(ptr %va1, ptr %va0)
  call void @llvm.va_end(ptr %va1)
  call void @llvm.lifetime.end.p0(i64 32, ptr %va1)
  call void @llvm.va_end(ptr %va0)
  call void @llvm.lifetime.end.p0(i64 32, ptr %va0)
  ret i32 0
}

