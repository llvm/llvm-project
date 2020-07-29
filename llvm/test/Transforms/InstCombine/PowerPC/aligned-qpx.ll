; RUN: opt -S -instcombine < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare <4 x double> @llvm.ppc.qpx.qvlfs(i8*) #1

define <4 x double> @test1(<4 x float>* %h) #0 {
entry:
  %h1 = getelementptr <4 x float>, <4 x float>* %h, i64 1
  %hv = bitcast <4 x float>* %h1 to i8*
  %vl = call <4 x double> @llvm.ppc.qpx.qvlfs(i8* %hv)

; CHECK-LABEL: @test1
; CHECK: @llvm.ppc.qpx.qvlfs
; CHECK: ret <4 x double>

  %v0 = load <4 x float>, <4 x float>* %h, align 8
  %v0e = fpext <4 x float> %v0 to <4 x double>
  %a = fadd <4 x double> %v0e, %vl
  ret <4 x double> %a
}

define <4 x double> @test1a(<4 x float>* align 16 %h) #0 {
entry:
  %h1 = getelementptr <4 x float>, <4 x float>* %h, i64 1
  %hv = bitcast <4 x float>* %h1 to i8*
  %vl = call <4 x double> @llvm.ppc.qpx.qvlfs(i8* %hv)

; CHECK-LABEL: @test1a
; CHECK-NOT: @llvm.ppc.qpx.qvlfs
; CHECK-NOT: load <4 x double>
; CHECK: ret <4 x double>

  %v0 = load <4 x float>, <4 x float>* %h, align 8
  %v0e = fpext <4 x float> %v0 to <4 x double>
  %a = fadd <4 x double> %v0e, %vl
  ret <4 x double> %a
}

declare void @llvm.ppc.qpx.qvstfs(<4 x double>, i8*) #0

define <4 x float> @test2(<4 x float>* %h, <4 x double> %d) #0 {
entry:
  %h1 = getelementptr <4 x float>, <4 x float>* %h, i64 1
  %hv = bitcast <4 x float>* %h1 to i8*
  call void @llvm.ppc.qpx.qvstfs(<4 x double> %d, i8* %hv)

  %v0 = load <4 x float>, <4 x float>* %h, align 8
  ret <4 x float> %v0

; CHECK-LABEL: @test2
; CHECK: @llvm.ppc.qpx.qvstfs
; CHECK: ret <4 x float>
}

define <4 x float> @test2a(<4 x float>* align 16 %h, <4 x double> %d) #0 {
entry:
  %h1 = getelementptr <4 x float>, <4 x float>* %h, i64 1
  %hv = bitcast <4 x float>* %h1 to i8*
  call void @llvm.ppc.qpx.qvstfs(<4 x double> %d, i8* %hv)

  %v0 = load <4 x float>, <4 x float>* %h, align 8
  ret <4 x float> %v0

; CHECK-LABEL: @test2
; CHECK: fptrunc <4 x double> %d to <4 x float>
; CHECK-NOT: @llvm.ppc.qpx.qvstfs
; CHECK-NOT: store <4 x double>
; CHECK: ret <4 x float>
}

declare <4 x double> @llvm.ppc.qpx.qvlfd(i8*) #1

define <4 x double> @test1l(<4 x double>* %h) #0 {
entry:
  %h1 = getelementptr <4 x double>, <4 x double>* %h, i64 1
  %hv = bitcast <4 x double>* %h1 to i8*
  %vl = call <4 x double> @llvm.ppc.qpx.qvlfd(i8* %hv)

; CHECK-LABEL: @test1l
; CHECK: @llvm.ppc.qpx.qvlfd
; CHECK: ret <4 x double>

  %v0 = load <4 x double>, <4 x double>* %h, align 8
  %a = fadd <4 x double> %v0, %vl
  ret <4 x double> %a
}

define <4 x double> @test1ln(<4 x double>* align 16 %h) #0 {
entry:
  %h1 = getelementptr <4 x double>, <4 x double>* %h, i64 1
  %hv = bitcast <4 x double>* %h1 to i8*
  %vl = call <4 x double> @llvm.ppc.qpx.qvlfd(i8* %hv)

; CHECK-LABEL: @test1ln
; CHECK: @llvm.ppc.qpx.qvlfd
; CHECK: ret <4 x double>

  %v0 = load <4 x double>, <4 x double>* %h, align 8
  %a = fadd <4 x double> %v0, %vl
  ret <4 x double> %a
}

define <4 x double> @test1la(<4 x double>* align 32 %h) #0 {
entry:
  %h1 = getelementptr <4 x double>, <4 x double>* %h, i64 1
  %hv = bitcast <4 x double>* %h1 to i8*
  %vl = call <4 x double> @llvm.ppc.qpx.qvlfd(i8* %hv)

; CHECK-LABEL: @test1la
; CHECK-NOT: @llvm.ppc.qpx.qvlfd
; CHECK: ret <4 x double>

  %v0 = load <4 x double>, <4 x double>* %h, align 8
  %a = fadd <4 x double> %v0, %vl
  ret <4 x double> %a
}

declare void @llvm.ppc.qpx.qvstfd(<4 x double>, i8*) #0

define <4 x double> @test2l(<4 x double>* %h, <4 x double> %d) #0 {
entry:
  %h1 = getelementptr <4 x double>, <4 x double>* %h, i64 1
  %hv = bitcast <4 x double>* %h1 to i8*
  call void @llvm.ppc.qpx.qvstfd(<4 x double> %d, i8* %hv)

  %v0 = load <4 x double>, <4 x double>* %h, align 8
  ret <4 x double> %v0

; CHECK-LABEL: @test2l
; CHECK: @llvm.ppc.qpx.qvstfd
; CHECK: ret <4 x double>
}

define <4 x double> @test2ln(<4 x double>* align 16 %h, <4 x double> %d) #0 {
entry:
  %h1 = getelementptr <4 x double>, <4 x double>* %h, i64 1
  %hv = bitcast <4 x double>* %h1 to i8*
  call void @llvm.ppc.qpx.qvstfd(<4 x double> %d, i8* %hv)

  %v0 = load <4 x double>, <4 x double>* %h, align 8
  ret <4 x double> %v0

; CHECK-LABEL: @test2ln
; CHECK: @llvm.ppc.qpx.qvstfd
; CHECK: ret <4 x double>
}

define <4 x double> @test2la(<4 x double>* align 32 %h, <4 x double> %d) #0 {
entry:
  %h1 = getelementptr <4 x double>, <4 x double>* %h, i64 1
  %hv = bitcast <4 x double>* %h1 to i8*
  call void @llvm.ppc.qpx.qvstfd(<4 x double> %d, i8* %hv)

  %v0 = load <4 x double>, <4 x double>* %h, align 8
  ret <4 x double> %v0

; CHECK-LABEL: @test2l
; CHECK-NOT: @llvm.ppc.qpx.qvstfd
; CHECK: ret <4 x double>
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }

