; RUN: llc < %s -mtriple=x86_64-- -o - | FileCheck %s

; This test verifies that we produce different code for different architectures
; based on target-cpu and target-features attributes.
; In this case avx has a vmovss instruction and otherwise we should be using movss
; to materialize constants.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define float @_Z3barv() #0 {
entry:
  ret float 4.000000e+00
}

; CHECK: barv
; CHECK: vmovss

define float @_Z4testv() #1 {
entry:
  ret float 1.000000e+00
}

; CHECK: testv
; CHECK: movss

define float @_Z3foov() #2 {
entry:
  ret float 4.000000e+00
}

; CHECK: foov
; CHECK: movss

define float @_Z3bazv() #0 {
entry:
  ret float 4.000000e+00
}

; CHECK: bazv
; CHECK: vmovss

define <2 x i64> @foo(<2 x i64> %a) #3 {
entry:
  %a.addr = alloca <2 x i64>, align 16
  store <2 x i64> %a, <2 x i64>* %a.addr, align 16
  %0 = load <2 x i64>, <2 x i64>* %a.addr, align 16
  %1 = call <2 x i64> @llvm.x86.aesni.aeskeygenassist(<2 x i64> %0, i8 4)
  ret <2 x i64> %1
}

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.x86.aesni.aeskeygenassist(<2 x i64>, i8)

; CHECK: foo
; CHECK: aeskeygenassist

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %crc, i8* %a) #3 {
entry:
  %crc.addr = alloca i32, align 4
  %a.addr = alloca i8*, align 8
  store i32 %crc, i32* %crc.addr, align 4
  store i8* %a, i8** %a.addr, align 8
  %0 = load i32, i32* %crc.addr, align 4
  %1 = load i8*, i8** %a.addr, align 8
  %incdec.ptr = getelementptr inbounds i8, i8* %1, i32 1
  store i8* %incdec.ptr, i8** %a.addr, align 8
  %2 = load i8, i8* %1, align 1
  %3 = call i32 @llvm.x86.sse42.crc32.32.8(i32 %0, i8 %2)
  ret i32 %3
}

; Function Attrs: nounwind readnone
declare i32 @llvm.x86.sse42.crc32.32.8(i32, i8)

; CHECK: bar
; CHECK: crc32b

attributes #0 = { "target-cpu"="x86-64" "target-features"="+avx2" }
attributes #1 = { "target-cpu"="x86-64" }
attributes #2 = { "target-cpu"="corei7" "target-features"="+sse4.2" }
attributes #3 = { "target-cpu"="x86-64" "target-features"="+avx2,+aes" }
