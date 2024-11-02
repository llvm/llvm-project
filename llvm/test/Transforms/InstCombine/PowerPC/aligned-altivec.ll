; RUN: opt -S -passes=instcombine < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare <4 x i32> @llvm.ppc.altivec.lvx(ptr) #1

define <4 x i32> @test1(ptr %h) #0 {
entry:
  %h1 = getelementptr <4 x i32>, ptr %h, i64 1
  %vl = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %h1)

; CHECK-LABEL: @test1
; CHECK: @llvm.ppc.altivec.lvx
; CHECK: ret <4 x i32>

  %v0 = load <4 x i32>, ptr %h, align 8
  %a = add <4 x i32> %v0, %vl
  ret <4 x i32> %a
}

define <4 x i32> @test1a(ptr align 16 %h) #0 {
entry:
  %h1 = getelementptr <4 x i32>, ptr %h, i64 1
  %vl = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %h1)

; CHECK-LABEL: @test1a
; CHECK-NOT: @llvm.ppc.altivec.lvx
; CHECK: ret <4 x i32>

  %v0 = load <4 x i32>, ptr %h, align 8
  %a = add <4 x i32> %v0, %vl
  ret <4 x i32> %a
}

declare void @llvm.ppc.altivec.stvx(<4 x i32>, ptr) #0

define <4 x i32> @test2(ptr %h, <4 x i32> %d) #0 {
entry:
  %h1 = getelementptr <4 x i32>, ptr %h, i64 1
  call void @llvm.ppc.altivec.stvx(<4 x i32> %d, ptr %h1)

  %v0 = load <4 x i32>, ptr %h, align 8
  ret <4 x i32> %v0

; CHECK-LABEL: @test2
; CHECK: @llvm.ppc.altivec.stvx
; CHECK: ret <4 x i32>
}

define <4 x i32> @test2a(ptr align 16 %h, <4 x i32> %d) #0 {
entry:
  %h1 = getelementptr <4 x i32>, ptr %h, i64 1
  call void @llvm.ppc.altivec.stvx(<4 x i32> %d, ptr %h1)

  %v0 = load <4 x i32>, ptr %h, align 8
  ret <4 x i32> %v0

; CHECK-LABEL: @test2
; CHECK-NOT: @llvm.ppc.altivec.stvx
; CHECK: ret <4 x i32>
}

declare <4 x i32> @llvm.ppc.altivec.lvxl(ptr) #1

define <4 x i32> @test1l(ptr %h) #0 {
entry:
  %h1 = getelementptr <4 x i32>, ptr %h, i64 1
  %vl = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %h1)

; CHECK-LABEL: @test1l
; CHECK: @llvm.ppc.altivec.lvxl
; CHECK: ret <4 x i32>

  %v0 = load <4 x i32>, ptr %h, align 8
  %a = add <4 x i32> %v0, %vl
  ret <4 x i32> %a
}

define <4 x i32> @test1la(ptr align 16 %h) #0 {
entry:
  %h1 = getelementptr <4 x i32>, ptr %h, i64 1
  %vl = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %h1)

; CHECK-LABEL: @test1la
; CHECK-NOT: @llvm.ppc.altivec.lvxl
; CHECK: ret <4 x i32>

  %v0 = load <4 x i32>, ptr %h, align 8
  %a = add <4 x i32> %v0, %vl
  ret <4 x i32> %a
}

declare void @llvm.ppc.altivec.stvxl(<4 x i32>, ptr) #0

define <4 x i32> @test2l(ptr %h, <4 x i32> %d) #0 {
entry:
  %h1 = getelementptr <4 x i32>, ptr %h, i64 1
  call void @llvm.ppc.altivec.stvxl(<4 x i32> %d, ptr %h1)

  %v0 = load <4 x i32>, ptr %h, align 8
  ret <4 x i32> %v0

; CHECK-LABEL: @test2l
; CHECK: @llvm.ppc.altivec.stvxl
; CHECK: ret <4 x i32>
}

define <4 x i32> @test2la(ptr align 16 %h, <4 x i32> %d) #0 {
entry:
  %h1 = getelementptr <4 x i32>, ptr %h, i64 1
  call void @llvm.ppc.altivec.stvxl(<4 x i32> %d, ptr %h1)

  %v0 = load <4 x i32>, ptr %h, align 8
  ret <4 x i32> %v0

; CHECK-LABEL: @test2l
; CHECK-NOT: @llvm.ppc.altivec.stvxl
; CHECK: ret <4 x i32>
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }

