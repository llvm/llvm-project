; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw,+avx512vl,+avxvnni,-avx512vnni < %s | FileCheck %s --check-prefix=AVXVNNI
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw,+avx512vl,+avxvnniint8,-avx10.2 < %s | FileCheck %s --check-prefix=AVXVNNIINT8
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw,+avx512vl,+avxvnniint16,-avx10.2 < %s | FileCheck %s --check-prefix=AVXVNNIINT16

; Targets with AVX512 registers but only 256-bit VEX/EVEX dot-product
; instructions should split 512-bit partial reductions to two 256-bit dot
; products instead of expanding them as scalarized zmm arithmetic.

define <16 x i32> @partial_reduce_sumla_i8_v16i32(<16 x i32> %acc, <64 x i8> %a, <64 x i8> %b) {
; AVXVNNI-LABEL: partial_reduce_sumla_i8_v16i32:
; AVXVNNI:       {vex} vpdpbusd
; AVXVNNI:       {vex} vpdpbusd
; AVXVNNI-NOT:   vpmaddwd
; AVXVNNI:       retq
  %a.zext = zext <64 x i8> %a to <64 x i32>
  %b.sext = sext <64 x i8> %b to <64 x i32>
  %mul = mul nsw <64 x i32> %a.zext, %b.sext
  %res = call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v64i32(<16 x i32> %acc, <64 x i32> %mul)
  ret <16 x i32> %res
}

define <16 x i32> @partial_reduce_smla_i16_v16i32(<16 x i32> %acc, <32 x i16> %a, <32 x i16> %b) {
; AVXVNNI-LABEL: partial_reduce_smla_i16_v16i32:
; AVXVNNI:       {vex} vpdpwssd
; AVXVNNI:       vpmaddwd
; AVXVNNI:       vpaddd
; AVXVNNI-NOT:   vpmulld
; AVXVNNI:       retq
  %a.sext = sext <32 x i16> %a to <32 x i32>
  %b.sext = sext <32 x i16> %b to <32 x i32>
  %mul = mul nsw <32 x i32> %a.sext, %b.sext
  %res = call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v32i32(<16 x i32> %acc, <32 x i32> %mul)
  ret <16 x i32> %res
}

define <16 x i32> @partial_reduce_smla_i8_v16i32(<16 x i32> %acc, <64 x i8> %a, <64 x i8> %b) {
; AVXVNNIINT8-LABEL: partial_reduce_smla_i8_v16i32:
; AVXVNNIINT8:       vpdpbssd
; AVXVNNIINT8:       vpdpbssd
; AVXVNNIINT8-NOT:   vpmulld
; AVXVNNIINT8:       retq
  %a.sext = sext <64 x i8> %a to <64 x i32>
  %b.sext = sext <64 x i8> %b to <64 x i32>
  %mul = mul nsw <64 x i32> %a.sext, %b.sext
  %res = call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v64i32(<16 x i32> %acc, <64 x i32> %mul)
  ret <16 x i32> %res
}

define <16 x i32> @partial_reduce_umla_i8_v16i32(<16 x i32> %acc, <64 x i8> %a, <64 x i8> %b) {
; AVXVNNIINT8-LABEL: partial_reduce_umla_i8_v16i32:
; AVXVNNIINT8:       vpdpbuud
; AVXVNNIINT8:       vpdpbuud
; AVXVNNIINT8-NOT:   vpmaddwd
; AVXVNNIINT8:       retq
  %a.zext = zext <64 x i8> %a to <64 x i32>
  %b.zext = zext <64 x i8> %b to <64 x i32>
  %mul = mul nsw <64 x i32> %a.zext, %b.zext
  %res = call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v64i32(<16 x i32> %acc, <64 x i32> %mul)
  ret <16 x i32> %res
}

define <16 x i32> @partial_reduce_sumla_i16_v16i32(<16 x i32> %acc, <32 x i16> %a, <32 x i16> %b) {
; AVXVNNIINT16-LABEL: partial_reduce_sumla_i16_v16i32:
; AVXVNNIINT16:       vpdpwsud
; AVXVNNIINT16:       vpdpwsud
; AVXVNNIINT16-NOT:   vpmulld
; AVXVNNIINT16:       retq
  %a.sext = sext <32 x i16> %a to <32 x i32>
  %b.zext = zext <32 x i16> %b to <32 x i32>
  %mul = mul nsw <32 x i32> %a.sext, %b.zext
  %res = call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v32i32(<16 x i32> %acc, <32 x i32> %mul)
  ret <16 x i32> %res
}

define <16 x i32> @partial_reduce_umla_i16_v16i32(<16 x i32> %acc, <32 x i16> %a, <32 x i16> %b) {
; AVXVNNIINT16-LABEL: partial_reduce_umla_i16_v16i32:
; AVXVNNIINT16:       vpdpwuud
; AVXVNNIINT16:       vpdpwuud
; AVXVNNIINT16-NOT:   vpmulld
; AVXVNNIINT16:       retq
  %a.zext = zext <32 x i16> %a to <32 x i32>
  %b.zext = zext <32 x i16> %b to <32 x i32>
  %mul = mul nsw <32 x i32> %a.zext, %b.zext
  %res = call <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v32i32(<16 x i32> %acc, <32 x i32> %mul)
  ret <16 x i32> %res
}

declare <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v64i32(<16 x i32>, <64 x i32>)
declare <16 x i32> @llvm.vector.partial.reduce.add.v16i32.v32i32(<16 x i32>, <32 x i32>)
