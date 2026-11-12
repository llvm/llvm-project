; RUN: opt -passes=globalopt %s -S -o - | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { %class.Wrapper }
%class.Wrapper = type { i32 }

$Wrapper = comdat any

@kA = internal global %struct.A zeroinitializer, align 4
; CHECK: @kA = internal unnamed_addr constant %struct.A { %class.Wrapper { i32 1036831949 } }, align 4

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } {
i32 65535, ptr @_GLOBAL__sub_I_const_static.cc, ptr null }]

define dso_local i32 @AsBits(ptr %x) #0 {
entry:
  %0 = load i32, ptr %x, align 4
  ret i32 %0
}

define internal void @__cxx_global_var_init() #1 section ".text.startup" {
entry:
  call void @Wrapper(ptr @kA, float 0x3FB99999A0000000)
  %0 = call ptr @llvm.invariant.start.p0(i64 4, ptr @kA)
  ret void
}

define linkonce_odr dso_local void @Wrapper(ptr %this, float %x) unnamed_addr #0 comdat align 2 {
entry:
  %x.addr = alloca float, align 4
  store float %x, ptr %x.addr, align 4
  %call = call i32 @AsBits(ptr %x.addr)
  store i32 %call, ptr %this, align 4
  ret void
}

declare ptr @llvm.invariant.start.p0(i64, ptr nocapture) #2

; Function Attrs: nounwind uwtable
define dso_local void @LoadIt(ptr %c) #0 {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %c, ptr align 4 @kA, i64 4, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #2

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_const_static.cc() #1 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" }
attributes #1 = { uwtable "target-cpu"="x86-64" }
attributes #2 = { argmemonly nounwind }
