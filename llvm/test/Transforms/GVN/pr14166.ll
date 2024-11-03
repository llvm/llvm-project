; RUN: opt -disable-basic-aa -passes=gvn -S < %s | FileCheck %s
target datalayout = "e-p:32:32:32"
target triple = "i386-pc-linux-gnu"
define <2 x i32> @test1() {
  %v1 = alloca <2 x i32>
  call void @anything(ptr %v1)
  %v2 = load <2 x i32>, ptr %v1
  %v3 = inttoptr <2 x i32> %v2 to <2 x ptr>
  store <2 x ptr> %v3, ptr %v1
  %v5 = load <2 x i32>, ptr %v1
  ret <2 x i32> %v5
; CHECK-LABEL: @test1(
; CHECK: %v1 = alloca <2 x i32>
; CHECK: call void @anything(ptr %v1)
; CHECK: %v2 = load <2 x i32>, ptr %v1
; CHECK: %v3 = inttoptr <2 x i32> %v2 to <2 x ptr>
; CHECK: store <2 x ptr> %v3, ptr %v1
; CHECK: ret <2 x i32> %v2
}

declare void @anything(ptr)

