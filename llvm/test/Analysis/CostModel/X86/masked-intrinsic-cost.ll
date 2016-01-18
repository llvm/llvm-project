; RUN: opt -S -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -cost-model -analyze < %s | FileCheck %s -check-prefix=AVX2


; AVX2-LABEL: test1
; AVX2: Found an estimated cost of 4 {{.*}}.masked
define <2 x double> @test1(<2 x i64> %trigger, <2 x double>* %addr, <2 x double> %dst) {
  %mask = icmp eq <2 x i64> %trigger, zeroinitializer
  %res = call <2 x double> @llvm.masked.load.v2f64(<2 x double>* %addr, i32 4, <2 x i1>%mask, <2 x double>%dst)
  ret <2 x double> %res
}

; AVX2-LABEL: test2
; AVX2: Found an estimated cost of 4 {{.*}}.masked
define <4 x i32> @test2(<4 x i32> %trigger, <4 x i32>* %addr, <4 x i32> %dst) {
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x i32> @llvm.masked.load.v4i32(<4 x i32>* %addr, i32 4, <4 x i1>%mask, <4 x i32>%dst)
  ret <4 x i32> %res
}

; AVX2-LABEL: test3
; AVX2: Found an estimated cost of 4 {{.*}}.masked
define void @test3(<4 x i32> %trigger, <4 x i32>* %addr, <4 x i32> %val) {
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v4i32(<4 x i32>%val, <4 x i32>* %addr, i32 4, <4 x i1>%mask)
  ret void
}

; AVX2-LABEL: test4
; AVX2: Found an estimated cost of 4 {{.*}}.masked
define <8 x float> @test4(<8 x i32> %trigger, <8 x float>* %addr, <8 x float> %dst) {
  %mask = icmp eq <8 x i32> %trigger, zeroinitializer
  %res = call <8 x float> @llvm.masked.load.v8f32(<8 x float>* %addr, i32 4, <8 x i1>%mask, <8 x float>%dst)
  ret <8 x float> %res
}

; AVX2-LABEL: test5
; AVX2: Found an estimated cost of 5 {{.*}}.masked
define void @test5(<2 x i32> %trigger, <2 x float>* %addr, <2 x float> %val) {
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v2f32(<2 x float>%val, <2 x float>* %addr, i32 4, <2 x i1>%mask)
  ret void
}

; AVX2-LABEL: test6
; AVX2: Found an estimated cost of 6 {{.*}}.masked
define void @test6(<2 x i32> %trigger, <2 x i32>* %addr, <2 x i32> %val) {
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  call void @llvm.masked.store.v2i32(<2 x i32>%val, <2 x i32>* %addr, i32 4, <2 x i1>%mask)
  ret void
}

; AVX2-LABEL: test7
; AVX2: Found an estimated cost of 5 {{.*}}.masked
define <2 x float> @test7(<2 x i32> %trigger, <2 x float>* %addr, <2 x float> %dst) {
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  %res = call <2 x float> @llvm.masked.load.v2f32(<2 x float>* %addr, i32 4, <2 x i1>%mask, <2 x float>%dst)
  ret <2 x float> %res
}

; AVX2-LABEL: test8
; AVX2: Found an estimated cost of 6 {{.*}}.masked
define <2 x i32> @test8(<2 x i32> %trigger, <2 x i32>* %addr, <2 x i32> %dst) {
  %mask = icmp eq <2 x i32> %trigger, zeroinitializer
  %res = call <2 x i32> @llvm.masked.load.v2i32(<2 x i32>* %addr, i32 4, <2 x i1>%mask, <2 x i32>%dst)
  ret <2 x i32> %res
}


declare <16 x i32> @llvm.masked.load.v16i32(<16 x i32>*, i32, <16 x i1>, <16 x i32>)
declare <4 x i32> @llvm.masked.load.v4i32(<4 x i32>*, i32, <4 x i1>, <4 x i32>)
declare <2 x i32> @llvm.masked.load.v2i32(<2 x i32>*, i32, <2 x i1>, <2 x i32>)
declare void @llvm.masked.store.v16i32(<16 x i32>, <16 x i32>*, i32, <16 x i1>)
declare void @llvm.masked.store.v8i32(<8 x i32>, <8 x i32>*, i32, <8 x i1>)
declare void @llvm.masked.store.v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>)
declare void @llvm.masked.store.v2f32(<2 x float>, <2 x float>*, i32, <2 x i1>)
declare void @llvm.masked.store.v2i32(<2 x i32>, <2 x i32>*, i32, <2 x i1>)
declare void @llvm.masked.store.v16f32(<16 x float>, <16 x float>*, i32, <16 x i1>)
declare void @llvm.masked.store.v16f32p(<16 x float>*, <16 x float>**, i32, <16 x i1>)
declare <16 x float> @llvm.masked.load.v16f32(<16 x float>*, i32, <16 x i1>, <16 x float>)
declare <8 x float> @llvm.masked.load.v8f32(<8 x float>*, i32, <8 x i1>, <8 x float>)
declare <4 x float> @llvm.masked.load.v4f32(<4 x float>*, i32, <4 x i1>, <4 x float>)
declare <2 x float> @llvm.masked.load.v2f32(<2 x float>*, i32, <2 x i1>, <2 x float>)
declare <8 x double> @llvm.masked.load.v8f64(<8 x double>*, i32, <8 x i1>, <8 x double>)
declare <4 x double> @llvm.masked.load.v4f64(<4 x double>*, i32, <4 x i1>, <4 x double>)
declare <2 x double> @llvm.masked.load.v2f64(<2 x double>*, i32, <2 x i1>, <2 x double>)
declare void @llvm.masked.store.v8f64(<8 x double>, <8 x double>*, i32, <8 x i1>)
declare void @llvm.masked.store.v2f64(<2 x double>, <2 x double>*, i32, <2 x i1>)
declare void @llvm.masked.store.v2i64(<2 x i64>, <2 x i64>*, i32, <2 x i1>)

