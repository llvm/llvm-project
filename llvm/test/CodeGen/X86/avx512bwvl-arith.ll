; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw -mattr=+avx512vl| FileCheck %s

; 256-bit

; CHECK-LABEL: vpaddb256_test
; CHECK: vpaddb %ymm{{.*}}
; CHECK: ret
define <32 x i8> @vpaddb256_test(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = add <32 x i8> %i, %j
  ret <32 x i8> %x
}

; CHECK-LABEL: vpaddb256_fold_test
; CHECK: vpaddb (%rdi), %ymm{{.*}}
; CHECK: ret
define <32 x i8> @vpaddb256_fold_test(<32 x i8> %i, <32 x i8>* %j) nounwind {
  %tmp = load <32 x i8>, <32 x i8>* %j, align 4
  %x = add <32 x i8> %i, %tmp
  ret <32 x i8> %x
}

; CHECK-LABEL: vpaddw256_test
; CHECK: vpaddw %ymm{{.*}}
; CHECK: ret
define <16 x i16> @vpaddw256_test(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = add <16 x i16> %i, %j
  ret <16 x i16> %x
}

; CHECK-LABEL: vpaddw256_fold_test
; CHECK: vpaddw (%rdi), %ymm{{.*}}
; CHECK: ret
define <16 x i16> @vpaddw256_fold_test(<16 x i16> %i, <16 x i16>* %j) nounwind {
  %tmp = load <16 x i16>, <16 x i16>* %j, align 4
  %x = add <16 x i16> %i, %tmp
  ret <16 x i16> %x
}

; CHECK-LABEL: vpaddw256_mask_test
; CHECK: vpaddw %ymm{{.*%k[1-7].*}}
; CHECK: ret
define <16 x i16> @vpaddw256_mask_test(<16 x i16> %i, <16 x i16> %j, <16 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <16 x i16> %mask1, zeroinitializer
  %x = add <16 x i16> %i, %j
  %r = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %i
  ret <16 x i16> %r
}

; CHECK-LABEL: vpaddw256_maskz_test
; CHECK: vpaddw %ymm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <16 x i16> @vpaddw256_maskz_test(<16 x i16> %i, <16 x i16> %j, <16 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <16 x i16> %mask1, zeroinitializer
  %x = add <16 x i16> %i, %j
  %r = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> zeroinitializer
  ret <16 x i16> %r
}

; CHECK-LABEL: vpaddw256_mask_fold_test
; CHECK: vpaddw (%rdi), %ymm{{.*%k[1-7]}}
; CHECK: ret
define <16 x i16> @vpaddw256_mask_fold_test(<16 x i16> %i, <16 x i16>* %j.ptr, <16 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <16 x i16> %mask1, zeroinitializer
  %j = load <16 x i16>, <16 x i16>* %j.ptr
  %x = add <16 x i16> %i, %j
  %r = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> %i
  ret <16 x i16> %r
}

; CHECK-LABEL: vpaddw256_maskz_fold_test
; CHECK: vpaddw (%rdi), %ymm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <16 x i16> @vpaddw256_maskz_fold_test(<16 x i16> %i, <16 x i16>* %j.ptr, <16 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <16 x i16> %mask1, zeroinitializer
  %j = load <16 x i16>, <16 x i16>* %j.ptr
  %x = add <16 x i16> %i, %j
  %r = select <16 x i1> %mask, <16 x i16> %x, <16 x i16> zeroinitializer
  ret <16 x i16> %r
}

; CHECK-LABEL: vpsubb256_test
; CHECK: vpsubb %ymm{{.*}}
; CHECK: ret
define <32 x i8> @vpsubb256_test(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = sub <32 x i8> %i, %j
  ret <32 x i8> %x
}

; CHECK-LABEL: vpsubw256_test
; CHECK: vpsubw %ymm{{.*}}
; CHECK: ret
define <16 x i16> @vpsubw256_test(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = sub <16 x i16> %i, %j
  ret <16 x i16> %x
}

; CHECK-LABEL: vpmullw256_test
; CHECK: vpmullw %ymm{{.*}}
; CHECK: ret
define <16 x i16> @vpmullw256_test(<16 x i16> %i, <16 x i16> %j) {
  %x = mul <16 x i16> %i, %j
  ret <16 x i16> %x
}

; 128-bit

; CHECK-LABEL: vpaddb128_test
; CHECK: vpaddb %xmm{{.*}}
; CHECK: ret
define <16 x i8> @vpaddb128_test(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %x = add <16 x i8> %i, %j
  ret <16 x i8> %x
}

; CHECK-LABEL: vpaddb128_fold_test
; CHECK: vpaddb (%rdi), %xmm{{.*}}
; CHECK: ret
define <16 x i8> @vpaddb128_fold_test(<16 x i8> %i, <16 x i8>* %j) nounwind {
  %tmp = load <16 x i8>, <16 x i8>* %j, align 4
  %x = add <16 x i8> %i, %tmp
  ret <16 x i8> %x
}

; CHECK-LABEL: vpaddw128_test
; CHECK: vpaddw %xmm{{.*}}
; CHECK: ret
define <8 x i16> @vpaddw128_test(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %x = add <8 x i16> %i, %j
  ret <8 x i16> %x
}

; CHECK-LABEL: vpaddw128_fold_test
; CHECK: vpaddw (%rdi), %xmm{{.*}}
; CHECK: ret
define <8 x i16> @vpaddw128_fold_test(<8 x i16> %i, <8 x i16>* %j) nounwind {
  %tmp = load <8 x i16>, <8 x i16>* %j, align 4
  %x = add <8 x i16> %i, %tmp
  ret <8 x i16> %x
}

; CHECK-LABEL: vpaddw128_mask_test
; CHECK: vpaddw %xmm{{.*%k[1-7].*}}
; CHECK: ret
define <8 x i16> @vpaddw128_mask_test(<8 x i16> %i, <8 x i16> %j, <8 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i16> %mask1, zeroinitializer
  %x = add <8 x i16> %i, %j
  %r = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %i
  ret <8 x i16> %r
}

; CHECK-LABEL: vpaddw128_maskz_test
; CHECK: vpaddw %xmm{{.*{%k[1-7]} {z}.*}}
; CHECK: ret
define <8 x i16> @vpaddw128_maskz_test(<8 x i16> %i, <8 x i16> %j, <8 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i16> %mask1, zeroinitializer
  %x = add <8 x i16> %i, %j
  %r = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> zeroinitializer
  ret <8 x i16> %r
}

; CHECK-LABEL: vpaddw128_mask_fold_test
; CHECK: vpaddw (%rdi), %xmm{{.*%k[1-7]}}
; CHECK: ret
define <8 x i16> @vpaddw128_mask_fold_test(<8 x i16> %i, <8 x i16>* %j.ptr, <8 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i16> %mask1, zeroinitializer
  %j = load <8 x i16>, <8 x i16>* %j.ptr
  %x = add <8 x i16> %i, %j
  %r = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> %i
  ret <8 x i16> %r
}

; CHECK-LABEL: vpaddw128_maskz_fold_test
; CHECK: vpaddw (%rdi), %xmm{{.*{%k[1-7]} {z}}}
; CHECK: ret
define <8 x i16> @vpaddw128_maskz_fold_test(<8 x i16> %i, <8 x i16>* %j.ptr, <8 x i16> %mask1) nounwind readnone {
  %mask = icmp ne <8 x i16> %mask1, zeroinitializer
  %j = load <8 x i16>, <8 x i16>* %j.ptr
  %x = add <8 x i16> %i, %j
  %r = select <8 x i1> %mask, <8 x i16> %x, <8 x i16> zeroinitializer
  ret <8 x i16> %r
}

; CHECK-LABEL: vpsubb128_test
; CHECK: vpsubb %xmm{{.*}}
; CHECK: ret
define <16 x i8> @vpsubb128_test(<16 x i8> %i, <16 x i8> %j) nounwind readnone {
  %x = sub <16 x i8> %i, %j
  ret <16 x i8> %x
}

; CHECK-LABEL: vpsubw128_test
; CHECK: vpsubw %xmm{{.*}}
; CHECK: ret
define <8 x i16> @vpsubw128_test(<8 x i16> %i, <8 x i16> %j) nounwind readnone {
  %x = sub <8 x i16> %i, %j
  ret <8 x i16> %x
}

; CHECK-LABEL: vpmullw128_test
; CHECK: vpmullw %xmm{{.*}}
; CHECK: ret
define <8 x i16> @vpmullw128_test(<8 x i16> %i, <8 x i16> %j) {
  %x = mul <8 x i16> %i, %j
  ret <8 x i16> %x
}

