; RUN: opt -passes=globalopt,instcombine %s -S -o - | FileCheck %s

; Static constructor should have been optimized out
; CHECK:       i32 @main
; CHECK-NEXT:     ret i32 69905
; CHECK-NOT:   _GLOBAL__sub_I_main.cpp

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.S = type { ptr }
%struct.A = type { i64, i64 }

@s = internal local_unnamed_addr global %struct.S zeroinitializer, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_main.cpp, ptr null }]
@gA = available_externally dso_local local_unnamed_addr global ptr inttoptr (i64 69905 to ptr), align 8

define dso_local i32 @main() local_unnamed_addr {
  %1 = load i64, ptr @s, align 8
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

define internal void @_GLOBAL__sub_I_main.cpp() section ".text.startup" {
  %1 = load i64, ptr @gA, align 8
  store i64 %1, ptr @s, align 8
  ret void
}
