; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=future < %s | \
; RUN:   FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=future < %s | \
; RUN:   FileCheck %s

define <4 x i32> @testLXVRL(ptr %a, i64 %b) {
; CHECK-LABEL: testLXVRL:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxvrl v2, r3, r4
; CHECK-NEXT:    blr
entry:
  %0 = tail call <4 x i32> @llvm.ppc.vsx.lxvrl(ptr %a, i64 %b)
  ret <4 x i32> %0
}
declare <4 x i32> @llvm.ppc.vsx.lxvrl(ptr, i64)

define <4 x i32> @testLXVRLL(ptr %a, i64 %b) {
; CHECK-LABEL: testLXVRLL:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxvrll v2, r3, r4
; CHECK-NEXT:    blr
entry:
  %0 = tail call <4 x i32> @llvm.ppc.vsx.lxvrll(ptr %a, i64 %b)
  ret <4 x i32> %0
}
declare <4 x i32> @llvm.ppc.vsx.lxvrll(ptr, i64)

define <256 x i1> @testLXVPRL(ptr %vpp, i64 %b) {
; CHECK-LABEL: testLXVPRL:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxvprl vsp34, r4, r5
; CHECK:         blr
entry:
  %0 = tail call <256 x i1> @llvm.ppc.vsx.lxvprl(ptr %vpp, i64 %b)
  ret <256 x i1> %0
}
declare <256 x i1> @llvm.ppc.vsx.lxvprl(ptr, i64)

define <256 x i1> @testLXVPRLL(ptr %vpp, i64 %b) {
; CHECK-LABEL: testLXVPRLL:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxvprll vsp34, r4, r5
; CHECK:         blr
entry:
  %0 = tail call <256 x i1> @llvm.ppc.vsx.lxvprll(ptr %vpp, i64 %b)
  ret <256 x i1> %0
}
declare <256 x i1> @llvm.ppc.vsx.lxvprll(ptr, i64)

define void @testSTXVRL(<4 x i32> %a, ptr %b, i64 %c) {
; CHECK-LABEL: testSTXVRL:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    stxvrl v2, [[REG:r[0-9]+]], [[REG1:r[0-9]+]]
; CHECK:         blr
entry:
  tail call void @llvm.ppc.vsx.stxvrl(<4 x i32> %a, ptr %b, i64 %c)
  ret void
}
declare void @llvm.ppc.vsx.stxvrl(<4 x i32>, ptr, i64)

define void @testSTXVRLL(<4 x i32> %a, ptr %b, i64 %c) {
; CHECK-LABEL: testSTXVRLL:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    stxvrll v2, [[REG:r[0-9]+]], [[REG1:r[0-9]+]]
; CHECK:         blr
entry:
  tail call void @llvm.ppc.vsx.stxvrll(<4 x i32> %a, ptr %b, i64 %c)
  ret void
}
declare void @llvm.ppc.vsx.stxvrll(<4 x i32>, ptr, i64)
