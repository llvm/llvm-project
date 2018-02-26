; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr9 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 < %s | FileCheck %s

define <8 x i16> @testXXBRH(<8 x i16> %a) {
; CHECK-LABEL: testXXBRH:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xxbrh 34, 34
; CHECK-NEXT:    blr

entry:
  %0 = bitcast <8 x i16> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
  %2 = bitcast <16 x i8> %1 to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @testXXBRW(<4 x i32> %a) {
; CHECK-LABEL: testXXBRW:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xxbrw 34, 34
; CHECK-NEXT:    blr

entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
}

define <2 x double> @testXXBRD(<2 x double> %a) {
; CHECK-LABEL: testXXBRD:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xxbrd 34, 34
; CHECK-NEXT:    blr

entry:
  %0 = bitcast <2 x double> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
  %2 = bitcast <16 x i8> %1 to <2 x double>
  ret <2 x double> %2
}

define <1 x i128> @testXXBRQ(<1 x i128> %a) {
; CHECK-LABEL: testXXBRQ:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xxbrq 34, 34
; CHECK-NEXT:    blr

entry:
  %0 = bitcast <1 x i128> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %2 = bitcast <16 x i8> %1 to <1 x i128>
  ret <1 x i128> %2
}
