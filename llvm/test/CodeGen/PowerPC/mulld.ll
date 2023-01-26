; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s \
; RUN: --check-prefix=CHECK-ITIN
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s \
; RUN: --check-prefix=CHECK-ITIN
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s \
; RUN: --check-prefix=CHECK-ITIN

define void @bn_mul_comba8(ptr nocapture %r, ptr nocapture readonly %a, ptr nocapture readonly %b) {
; CHECK-LABEL: bn_mul_comba8:
; CHECK:    mulhdu
; CHECK-NEXT:    mulld
; CHECK:         mulhdu
; CHECK:         mulld
; CHECK-NEXT:    mulhdu


; CHECK-ITIN-LABEL: bn_mul_comba8:
; CHECK-ITIN:    mulhdu
; CHECK-ITIN-NEXT:    mulld
; CHECK-ITIN-NEXT:    mulhdu
; CHECK-ITIN-NEXT:    mulld
; CHECK-ITIN-NEXT:    mulhdu

  %1 = load i64, ptr %a, align 8
  %conv = zext i64 %1 to i128
  %2 = load i64, ptr %b, align 8
  %conv2 = zext i64 %2 to i128
  %mul = mul nuw i128 %conv2, %conv
  %shr = lshr i128 %mul, 64
  %agep = getelementptr inbounds i64, ptr %a, i64 1
  %3 = load i64, ptr %agep, align 8
  %conv14 = zext i64 %3 to i128
  %mul15 = mul nuw i128 %conv14, %conv
  %add17 = add i128 %mul15, %shr
  %shr19 = lshr i128 %add17, 64
  %conv20 = trunc i128 %shr19 to i64
  %bgep = getelementptr inbounds i64, ptr %b, i64 1
  %4 = load i64, ptr %bgep, align 8
  %conv28 = zext i64 %4 to i128
  %mul31 = mul nuw i128 %conv28, %conv2
  %conv32 = and i128 %add17, 18446744073709551615
  %add33 = add i128 %conv32, %mul31
  %shr35 = lshr i128 %add33, 64
  %conv36 = trunc i128 %shr35 to i64
  %add37 = add i64 %conv36, %conv20
  %cmp38 = icmp ult i64 %add37, %conv36
  %conv148 = zext i1 %cmp38 to i64
  store i64 %conv148, ptr %r, align 8
  ret void
}

