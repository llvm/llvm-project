; Check that we don't crash when processing declaration with type metadata
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-none-linux-gnu"

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

@_ZTVN3foo3barE = external dso_local unnamed_addr constant { [8 x i8*] }, align 8, !type !0

define i1 @call1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [3 x i8*]**
  %vtable = load [3 x i8*]*, [3 x i8*]** %vtableptr
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i1 (i8*)*
  %result = call i1 %fptr_casted(i8* %obj)
  ret i1 %result
}

!0 = !{i64 16, !"_ZTSN3foo3barE"}
