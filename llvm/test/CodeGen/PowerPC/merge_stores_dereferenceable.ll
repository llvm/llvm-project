; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; This code causes an assertion failure if dereferenceable flag is not properly set when in merging consecutive stores
; CHECK-LABEL: func:
; CHECK: lxvd2x [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOT: lxvd2x
; CHECK: stxvd2x [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}

define <2 x i64> @func(ptr %pdst) {
entry:
  %a = alloca [4 x i64], align 8
  %psrc1 = getelementptr inbounds i64, ptr %a, i64 1
  %d0 = load i64, ptr %a
  %d1 = load i64, ptr %psrc1
  %pdst1 = getelementptr inbounds i64, ptr %pdst, i64 1
  store i64 %d0, ptr %pdst, align 8
  store i64 %d1, ptr %pdst1, align 8
  %vec = load <2 x i64>, ptr %a
  ret <2 x i64> %vec
}

