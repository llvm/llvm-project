; Check that we don't crash when processing declaration with type metadata
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-none-linux-gnu"

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

@_ZTVN3foo3barE = external dso_local unnamed_addr constant { [8 x ptr] }, align 8, !type !0

define i1 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i1 %fptr(ptr %obj)
  ret i1 %result
}

!0 = !{i64 16, !"_ZTSN3foo3barE"}
