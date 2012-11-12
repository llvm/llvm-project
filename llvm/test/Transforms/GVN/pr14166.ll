; RUN: opt -gvn -S < %s | FileCheck %s
target datalayout = "e-p:32:32:32"
target triple = "i386-pc-linux-gnu"
define <2 x i32> @test1() {
  %v1 = alloca <2 x i32>
  call void @anything(<2 x i32>* %v1)
  %v2 = load <2 x i32>* %v1
  %v3 = inttoptr <2 x i32> %v2 to <2 x i8*>
  %v4 = bitcast <2 x i32>* %v1 to <2 x i8*>*
  store <2 x i8*> %v3, <2 x i8*>* %v4
  %v5 = load <2 x i32>* %v1
  ret <2 x i32> %v5
; CHECK: @test1
; CHECK: %v1 = alloca <2 x i32>
; CHECK: call void @anything(<2 x i32>* %v1)
; CHECK: %v2 = load <2 x i32>* %v1
; CHECK: %v3 = inttoptr <2 x i32> %v2 to <2 x i8*>
; CHECK: %v4 = bitcast <2 x i32>* %v1 to <2 x i8*>*
; CHECK: store <2 x i8*> %v3, <2 x i8*>* %v4
; CHECK: %1 = ptrtoint <2 x i8*> %v3 to <2 x i32>
; CHECK: %2 = bitcast <2 x i32> %1 to i64
; CHECK: %3 = bitcast i64 %2 to <2 x i32>
; CHECK: ret <2 x i32> %3
}

declare void @anything(<2 x i32>*)

