; RUN: opt -S -instcombine < %s | FileCheck %s
; ARM64 AES intrinsic variants

define <16 x i8> @combineXorAeseZeroARM64(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: @combineXorAeseZeroARM64(
; CHECK-NEXT:    %data.aes = tail call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %data, <16 x i8> %key)
; CHECK-NEXT:    ret <16 x i8> %data.aes
  %data.xor = xor <16 x i8> %data, %key
  %data.aes = tail call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %data.xor, <16 x i8> zeroinitializer)
  ret <16 x i8> %data.aes
}

define <16 x i8> @combineXorAeseNonZeroARM64(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: @combineXorAeseNonZeroARM64(
; CHECK-NEXT:    %data.xor = xor <16 x i8> %data, %key
; CHECK-NEXT:    %data.aes = tail call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %data.xor, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
; CHECK-NEXT:    ret <16 x i8> %data.aes
  %data.xor = xor <16 x i8> %data, %key
  %data.aes = tail call <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8> %data.xor, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  ret <16 x i8> %data.aes
}

define <16 x i8> @combineXorAesdZeroARM64(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: @combineXorAesdZeroARM64(
; CHECK-NEXT:    %data.aes = tail call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %data, <16 x i8> %key)
; CHECK-NEXT:    ret <16 x i8> %data.aes
  %data.xor = xor <16 x i8> %data, %key
  %data.aes = tail call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %data.xor, <16 x i8> zeroinitializer)
  ret <16 x i8> %data.aes
}

define <16 x i8> @combineXorAesdNonZeroARM64(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: @combineXorAesdNonZeroARM64(
; CHECK-NEXT:    %data.xor = xor <16 x i8> %data, %key
; CHECK-NEXT:    %data.aes = tail call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %data.xor, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
; CHECK-NEXT:    ret <16 x i8> %data.aes
  %data.xor = xor <16 x i8> %data, %key
  %data.aes = tail call <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8> %data.xor, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  ret <16 x i8> %data.aes
}

declare <16 x i8> @llvm.aarch64.crypto.aese(<16 x i8>, <16 x i8>) #0
declare <16 x i8> @llvm.aarch64.crypto.aesd(<16 x i8>, <16 x i8>) #0

