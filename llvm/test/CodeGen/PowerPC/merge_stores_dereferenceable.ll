; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; This code causes an assertion failure if dereferenceable flag is not properly set when in merging consecutive stores
; CHECK-LABEL: func:
; CHECK: lxvd2x [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOT: lxvd2x
; CHECK: stxvd2x [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}

define <2 x i64> @func(i64* %pdst) {
entry:
  %a = alloca [4 x i64], align 8
  %psrc0 = bitcast [4 x i64]* %a to i64*
  %psrc1 = getelementptr inbounds i64, i64* %psrc0, i64 1
  %d0 = load i64, i64* %psrc0
  %d1 = load i64, i64* %psrc1
  %pdst0 = getelementptr inbounds i64, i64* %pdst, i64 0
  %pdst1 = getelementptr inbounds i64, i64* %pdst, i64 1
  store i64 %d0, i64* %pdst0, align 8
  store i64 %d1, i64* %pdst1, align 8
  %psrcd = bitcast [4 x i64]* %a to <2 x i64>*
  %vec = load <2 x i64>, <2 x i64>* %psrcd
  ret <2 x i64> %vec
}

