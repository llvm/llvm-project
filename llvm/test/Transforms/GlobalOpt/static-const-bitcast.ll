; RUN: opt -globalopt %s -S -o - | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { %class.Wrapper }
%class.Wrapper = type { i32 }

$Wrapper = comdat any

@kA = internal global %struct.A zeroinitializer, align 4
; CHECK: @kA = internal unnamed_addr constant %struct.A { %class.Wrapper { i32 1036831949 } }, align 4

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } {
i32 65535, void ()* @_GLOBAL__sub_I_const_static.cc, i8* null }]

define dso_local i32 @AsBits(float* %x) #0 {
entry:
  %0 = bitcast float* %x to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

define internal void @__cxx_global_var_init() #1 section ".text.startup" {
entry:
  call void @Wrapper(%class.Wrapper* getelementptr inbounds (%struct.A, %struct.A* @kA, i32 0, i32 0), float 0x3FB99999A0000000)
  %0 = call {}* @llvm.invariant.start.p0i8(i64 4, i8* bitcast (%struct.A* @kA to i8*))
  ret void
}

define linkonce_odr dso_local void @Wrapper(%class.Wrapper* %this, float %x) unnamed_addr #0 comdat align 2 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %store_ = getelementptr inbounds %class.Wrapper, %class.Wrapper* %this, i32 0, i32 0
  %call = call i32 @AsBits(float* %x.addr)
  store i32 %call, i32* %store_, align 4
  ret void
}

declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: nounwind uwtable
define dso_local void @LoadIt(%struct.A* %c) #0 {
entry:
  %0 = bitcast %struct.A* %c to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 bitcast (%struct.A* @kA to i8*), i64 4, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_const_static.cc() #1 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" }
attributes #1 = { uwtable "target-cpu"="x86-64" }
attributes #2 = { argmemonly nounwind }
