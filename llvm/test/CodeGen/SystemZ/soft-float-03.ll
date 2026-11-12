; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -mattr=soft-float -O3 < %s | FileCheck %s
;
; Check that soft-float implies "-vector".

define <2 x i64> @f0(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f0:
; CHECK-NOT: vag
; CHECK-NOT: %v
  %res = add <2 x i64> %val1, %val2
  ret <2 x i64> %res
}
