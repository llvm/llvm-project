; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw| FileCheck %s

; CHECK-LABEL: vpaddb512_test
; CHECK: vpaddb %zmm{{.*}}
; CHECK: ret
define <64 x i8> @vpaddb512_test(<64 x i8> %i, <64 x i8> %j) nounwind readnone {
  %x = add <64 x i8> %i, %j
  ret <64 x i8> %x
}

; CHECK-LABEL: vpaddb512_fold_test
; CHECK: vpaddb (%rdi), %zmm{{.*}}
; CHECK: ret
define <64 x i8> @vpaddb512_fold_test(<64 x i8> %i, <64 x i8>* %j) nounwind {
  %tmp = load <64 x i8>, <64 x i8>* %j, align 4
  %x = add <64 x i8> %i, %tmp
  ret <64 x i8> %x
}

; CHECK-LABEL: vpaddw512_test
; CHECK: vpaddw %zmm{{.*}}
; CHECK: ret
define <32 x i16> @vpaddw512_test(<32 x i16> %i, <32 x i16> %j) nounwind readnone {
  %x = add <32 x i16> %i, %j
  ret <32 x i16> %x
}

; CHECK-LABEL: vpaddw512_fold_test
; CHECK: vpaddw (%rdi), %zmm{{.*}}
; CHECK: ret
define <32 x i16> @vpaddw512_fold_test(<32 x i16> %i, <32 x i16>* %j) nounwind {
  %tmp = load <32 x i16>, <32 x i16>* %j, align 4
  %x = add <32 x i16> %i, %tmp
  ret <32 x i16> %x
}

; CHECK-LABEL: vpaddw512_mask_test
; CHECK: vpaddw %zmm{{.*%k[1-7].*}}
; CHECK: ret
define <32 x i16> @vpaddw512_mask_test(<32 x i16> %i, <32 x i16> %j, <32 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <32 x i16> %mask1, zeroinitializer
  %x = add <32 x i16> %i, %j
  %r = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> %i
  ret <32 x i16> %r
}

; CHECK-LABEL: vpaddw512_maskz_test
; CHECK: vpaddw %zmm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <32 x i16> @vpaddw512_maskz_test(<32 x i16> %i, <32 x i16> %j, <32 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <32 x i16> %mask1, zeroinitializer
  %x = add <32 x i16> %i, %j
  %r = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> zeroinitializer
  ret <32 x i16> %r
}

; CHECK-LABEL: vpaddw512_mask_fold_test
; CHECK: vpaddw (%rdi), %zmm{{.*%k[1-7]}}
; CHECK: ret
define <32 x i16> @vpaddw512_mask_fold_test(<32 x i16> %i, <32 x i16>* %j.ptr, <32 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <32 x i16> %mask1, zeroinitializer
  %j = load <32 x i16>, <32 x i16>* %j.ptr
  %x = add <32 x i16> %i, %j
  %r = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> %i
  ret <32 x i16> %r
}

; CHECK-LABEL: vpaddw512_maskz_fold_test
; CHECK: vpaddw (%rdi), %zmm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <32 x i16> @vpaddw512_maskz_fold_test(<32 x i16> %i, <32 x i16>* %j.ptr, <32 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <32 x i16> %mask1, zeroinitializer
  %j = load <32 x i16>, <32 x i16>* %j.ptr
  %x = add <32 x i16> %i, %j
  %r = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> zeroinitializer
  ret <32 x i16> %r
}

; CHECK-LABEL: vpsubb512_test
; CHECK: vpsubb %zmm{{.*}}
; CHECK: ret
define <64 x i8> @vpsubb512_test(<64 x i8> %i, <64 x i8> %j) nounwind readnone {
  %x = sub <64 x i8> %i, %j
  ret <64 x i8> %x
}

; CHECK-LABEL: vpsubw512_test
; CHECK: vpsubw %zmm{{.*}}
; CHECK: ret
define <32 x i16> @vpsubw512_test(<32 x i16> %i, <32 x i16> %j) nounwind readnone {
  %x = sub <32 x i16> %i, %j
  ret <32 x i16> %x
}

; CHECK-LABEL: vpmullw512_test
; CHECK: vpmullw %zmm{{.*}}
; CHECK: ret
define <32 x i16> @vpmullw512_test(<32 x i16> %i, <32 x i16> %j) {
  %x = mul <32 x i16> %i, %j
  ret <32 x i16> %x
}

