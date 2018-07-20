; RUN: opt -globalopt -instcombine %s -S -o - | FileCheck %s

; Static constructor should have been optimized out
; CHECK:       i32 @main
; CHECK-NEXT:     ret i32 69905
; CHECK-NOT:   _GLOBAL__sub_I_main.cpp

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.S = type { %struct.A* }
%struct.A = type { i64, i64 }

@s = internal local_unnamed_addr global %struct.S zeroinitializer, align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_main.cpp, i8* null }]
@gA = available_externally dso_local local_unnamed_addr global %struct.A* inttoptr (i64 69905 to %struct.A*), align 8

define dso_local i32 @main() local_unnamed_addr {
  %1 = load i64, i64* bitcast (%struct.S* @s to i64*), align 8
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

define internal void @_GLOBAL__sub_I_main.cpp() section ".text.startup" {
  %1 = load i64, i64* bitcast (%struct.A** @gA to i64*), align 8
  store i64 %1, i64* bitcast (%struct.S* @s to i64*), align 8
  ret void
}
