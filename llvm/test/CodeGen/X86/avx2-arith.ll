; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: vpaddq %ymm
define <4 x i64> @vpaddq(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = add <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK: vpaddd %ymm
define <8 x i32> @vpaddd(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = add <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK: vpaddw %ymm
define <16 x i16> @vpaddw(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = add <16 x i16> %i, %j
  ret <16 x i16> %x
}

; CHECK: vpaddb %ymm
define <32 x i8> @vpaddb(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = add <32 x i8> %i, %j
  ret <32 x i8> %x
}

; CHECK: vpsubq %ymm
define <4 x i64> @vpsubq(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = sub <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK: vpsubd %ymm
define <8 x i32> @vpsubd(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = sub <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK: vpsubw %ymm
define <16 x i16> @vpsubw(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = sub <16 x i16> %i, %j
  ret <16 x i16> %x
}

; CHECK: vpsubb %ymm
define <32 x i8> @vpsubb(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = sub <32 x i8> %i, %j
  ret <32 x i8> %x
}

; CHECK: vpmulld %ymm
define <8 x i32> @vpmulld(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = mul <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK: vpmullw %ymm
define <16 x i16> @vpmullw(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = mul <16 x i16> %i, %j
  ret <16 x i16> %x
}

; CHECK: vpmuludq %ymm
; CHECK-NEXT: vpsrlq $32, %ymm
; CHECK-NEXT: vpmuludq %ymm
; CHECK-NEXT: vpsllq $32, %ymm
; CHECK-NEXT: vpaddq %ymm
; CHECK-NEXT: vpsrlq $32, %ymm
; CHECK-NEXT: vpmuludq %ymm
; CHECK-NEXT: vpsllq $32, %ymm
; CHECK-NEXT: vpaddq %ymm
define <4 x i64> @mul-v4i64(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = mul <4 x i64> %i, %j
  ret <4 x i64> %x
}

