; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -verify-machineinstrs -ppc-asm-full-reg-names \
; RUN:   -ppc-vsr-nums-as-vr < %s | FileCheck %s

@best8x8mode = external dso_local local_unnamed_addr global [4 x i16], align 2
define dso_local void @AlignDSForm() local_unnamed_addr {
entry:
  %0 = load <4 x i16>, <4 x i16>* bitcast ([4 x i16]* @best8x8mode to <4 x i16>*), align 2
  store <4 x i16> %0, <4 x i16>* undef, align 4
  unreachable
; CHECK-LABEL: AlignDSForm
; CHECK: addis r{{[0-9]+}}, r{{[0-9]+}}, best8x8mode@toc@ha
; CHECK: addi r[[REG:[0-9]+]], r{{[0-9]+}}, best8x8mode@toc@l
; CHECK: ldx r{{[0-9]+}}, 0, r[[REG]]
}

