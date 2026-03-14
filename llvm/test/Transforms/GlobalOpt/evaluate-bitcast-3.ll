; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; Test the evaluation of a load via a bitcast and a store via a GEP.
; Check that globals are constant folded to the correct value.

; CHECK: @u = dso_local local_unnamed_addr global %union.A { ptr inttoptr (i64 12345 to ptr) }, align 8
; CHECK: @l = dso_local local_unnamed_addr global i64 12345, align 8

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%union.A = type { ptr }

$_ZN1AC2Ex = comdat any

@u = dso_local global %union.A zeroinitializer, align 8
@l = dso_local global i64 0, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_test.cpp, ptr null }]

define internal void @__cxx_global_var_init() section ".text.startup" {
  call void @_ZN1AC2Ex(ptr @u, i64 12345)
  ret void
}

define linkonce_odr dso_local void @_ZN1AC2Ex(ptr %this, i64 %ll) unnamed_addr comdat align 2 {
  %l = inttoptr i64 %ll to ptr
  store ptr %l, ptr %this
  ret void
}

define internal void @__cxx_global_var_init.1() section ".text.startup" {
  %1 = load i64, ptr @u, align 8
  store i64 %1, ptr @l, align 8
  ret void
}

define internal void @_GLOBAL__sub_I_test.cpp() section ".text.startup" {
  call void @__cxx_global_var_init()
  call void @__cxx_global_var_init.1()
  ret void
}
