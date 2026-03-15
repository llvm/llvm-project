; RUN: llc -mcpu=pwr10 -mtriple=powerpc64-ibm-aix -ppc-asm-full-reg-names \
; RUN:     -ppc-vsr-nums-as-vr < %s | FileCheck %s
; RUN: llc -mcpu=pwr10 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s
; RUN: llc -mcpu=pwr10 -mtriple=powerpc-ibm-aix -ppc-asm-full-reg-names \
; RUN:     -ppc-vsr-nums-as-vr < %s | FileCheck %s

define signext i32 @cmpgtw(<4 x i32> noundef %a, <4 x i32> noundef %b) local_unnamed_addr {
; CHECK: vcmpgtsw. v2, v2, v3
; CHECK: setbc r3, 4*cr6+lt
entry:
  %0 = tail call i32 @llvm.ppc.altivec.vcmpgtsw.p(i32 2, <4 x i32> %a, <4 x i32> %b)
  ret i32 %0
}

define signext i32 @cmpanynew(<4 x i32> noundef %a, <4 x i32> noundef %b) local_unnamed_addr {
; CHECK: vcmpequw. v2, v2, v3
; CHECK: setbcr r3, 4*cr6+lt
entry:
  %0 = tail call i32 @llvm.ppc.altivec.vcmpequw.p(i32 3, <4 x i32> %a, <4 x i32> %b)
  ret i32 %0
}

define signext i32 @cmpallneh(<8 x i16> noundef %a, <8 x i16> noundef %b) local_unnamed_addr {
; CHECK: vcmpequh. v2, v2, v3
; CHECK: setbc r3, 4*cr6+eq
entry:
  %0 = tail call i32 @llvm.ppc.altivec.vcmpequh.p(i32 0, <8 x i16> %a, <8 x i16> %b)
  ret i32 %0
}

define signext i32 @cmpeqb(<16 x i8> noundef %a, <16 x i8> noundef %b) local_unnamed_addr {
; CHECK: vcmpequb. v2, v2, v3
; CHECK: setbcr r3, 4*cr6+eq
entry:
  %0 = tail call i32 @llvm.ppc.altivec.vcmpequb.p(i32 1, <16 x i8> %a, <16 x i8> %b)
  ret i32 %0
}

declare i32 @llvm.ppc.altivec.vcmpgtsw.p(i32, <4 x i32>, <4 x i32>)

declare i32 @llvm.ppc.altivec.vcmpequw.p(i32, <4 x i32>, <4 x i32>)

declare i32 @llvm.ppc.altivec.vcmpequh.p(i32, <8 x i16>, <8 x i16>)

declare i32 @llvm.ppc.altivec.vcmpequb.p(i32, <16 x i8>, <16 x i8>)
