; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; Check handling of bswap with unsupported sizes.

declare i8 @llvm.bswap.i8(i8)
declare <2 x i8> @llvm.bswap.v2i8(<2 x i8>)

declare i12 @llvm.bswap.i12(i12)
declare <2 x i12> @llvm.bswap.v2i12(<2 x i12>)

declare i18 @llvm.bswap.i18(i18)
declare <2 x i18> @llvm.bswap.v2i18(<2 x i18>)

define i8 @bswap_i8(i8 %arg) {
; CHECK: bswap must be an even number of bytes
; CHECK-NEXT: %res = call i8 @llvm.bswap.i8(i8 %arg)
  %res = call i8 @llvm.bswap.i8(i8 %arg)
  ret i8 %res
}

define <2 x i8> @bswap_v2i8(<2 x i8> %arg) {
; CHECK: bswap must be an even number of bytes
; CHECK-NEXT: %res = call <2 x i8> @llvm.bswap.v2i8(<2 x i8> %arg)
  %res = call <2 x i8> @llvm.bswap.v2i8(<2 x i8> %arg)
  ret <2 x i8> %res
}

define i12 @bswap_i12(i12 %arg) {
; CHECK: bswap must be an even number of bytes
; CHECK-NEXT: %res = call i12 @llvm.bswap.i12(i12 %arg)
  %res = call i12 @llvm.bswap.i12(i12 %arg)
  ret i12 %res
}

define <2 x i12> @bswap_v2i12(<2 x i12> %arg) {
; CHECK: bswap must be an even number of bytes
; CHECK-NEXT: %res = call <2 x i12> @llvm.bswap.v2i12(<2 x i12> %arg)
  %res = call <2 x i12> @llvm.bswap.v2i12(<2 x i12> %arg)
  ret <2 x i12> %res
}

define i18 @bswap_i18(i18 %arg) {
; CHECK: bswap must be an even number of bytes
; CHECK-NEXT: %res = call i18 @llvm.bswap.i18(i18 %arg)
  %res = call i18 @llvm.bswap.i18(i18 %arg)
  ret i18 %res
}

define <2 x i18> @bswap_v2i18(<2 x i18> %arg) {
; CHECK: bswap must be an even number of bytes
; CHECK-NEXT: %res = call <2 x i18> @llvm.bswap.v2i18(<2 x i18> %arg)
  %res = call <2 x i18> @llvm.bswap.v2i18(<2 x i18> %arg)
  ret <2 x i18> %res
}
