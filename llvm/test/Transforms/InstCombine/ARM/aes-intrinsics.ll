; RUN: opt -S -instcombine < %s | FileCheck %s
; ARM AES intrinsic variants

define <16 x i8> @combineXorAeseZeroARM(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: @combineXorAeseZeroARM(
; CHECK-NEXT:    %data.aes = tail call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %data, <16 x i8> %key)
; CHECK-NEXT:    ret <16 x i8> %data.aes
  %data.xor = xor <16 x i8> %data, %key
  %data.aes = tail call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %data.xor, <16 x i8> zeroinitializer)
  ret <16 x i8> %data.aes
}

define <16 x i8> @combineXorAeseNonZeroARM(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: @combineXorAeseNonZeroARM(
; CHECK-NEXT:    %data.xor = xor <16 x i8> %data, %key
; CHECK-NEXT:    %data.aes = tail call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %data.xor, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
; CHECK-NEXT:    ret <16 x i8> %data.aes
  %data.xor = xor <16 x i8> %data, %key
  %data.aes = tail call <16 x i8> @llvm.arm.neon.aese(<16 x i8> %data.xor, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  ret <16 x i8> %data.aes
}

define <16 x i8> @combineXorAesdZeroARM(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: @combineXorAesdZeroARM(
; CHECK-NEXT:    %data.aes = tail call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %data, <16 x i8> %key)
; CHECK-NEXT:    ret <16 x i8> %data.aes
  %data.xor = xor <16 x i8> %data, %key
  %data.aes = tail call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %data.xor, <16 x i8> zeroinitializer)
  ret <16 x i8> %data.aes
}

define <16 x i8> @combineXorAesdNonZeroARM(<16 x i8> %data, <16 x i8> %key) {
; CHECK-LABEL: @combineXorAesdNonZeroARM(
; CHECK-NEXT:    %data.xor = xor <16 x i8> %data, %key
; CHECK-NEXT:    %data.aes = tail call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %data.xor, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
; CHECK-NEXT:    ret <16 x i8> %data.aes
  %data.xor = xor <16 x i8> %data, %key
  %data.aes = tail call <16 x i8> @llvm.arm.neon.aesd(<16 x i8> %data.xor, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  ret <16 x i8> %data.aes
}

declare <16 x i8> @llvm.arm.neon.aese(<16 x i8>, <16 x i8>) #0
declare <16 x i8> @llvm.arm.neon.aesd(<16 x i8>, <16 x i8>) #0
