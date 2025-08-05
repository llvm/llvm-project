; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=future < %s | \
; RUN:   FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=future < %s | \
; RUN:   FileCheck %s

; Test for load/store to/from v4i32.

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

; Test for load/store to/from v2i64.

define <2 x i64> @testLXVRL2(ptr %a, i64 %b) {
; CHECK-LABEL: testLXVRL2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxvrl v2, r3, r4
; CHECK-NEXT:    blr
entry:
  %0 = tail call <2 x i64> @llvm.ppc.vsx.lxvrl.v2i64(ptr %a, i64 %b)
  ret <2 x i64> %0
}
declare <2 x i64> @llvm.ppc.vsx.lxvrl.v2i64(ptr, i64)

define <2 x i64> @testLXVRLL2(ptr %a, i64 %b) {
; CHECK-LABEL: testLXVRLL2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxvrll v2, r3, r4
; CHECK-NEXT:    blr
entry:
  %0 = tail call <2 x i64> @llvm.ppc.vsx.lxvrll.v2i64(ptr %a, i64 %b)
  ret <2 x i64> %0
}
declare <2 x i64> @llvm.ppc.vsx.lxvrll.v2i64(ptr, i64)

define void @testSTXVRL2(<2 x i64> %a, ptr %b, i64 %c) {
; CHECK-LABEL: testSTXVRL2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    stxvrl v2, [[REG:r[0-9]+]], [[REG1:r[0-9]+]]
; CHECK:         blr
entry:
  tail call void @llvm.ppc.vsx.stxvrl.v2i64(<2 x i64> %a, ptr %b, i64 %c)
  ret void
}
declare void @llvm.ppc.vsx.stxvrl.v2i64(<2 x i64>, ptr, i64)

define void @testSTXVRLL2(<2 x i64> %a, ptr %b, i64 %c) {
; CHECK-LABEL: testSTXVRLL2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    stxvrll v2, [[REG:r[0-9]+]], [[REG1:r[0-9]+]]
; CHECK:         blr
entry:
  tail call void @llvm.ppc.vsx.stxvrll.v2i64(<2 x i64> %a, ptr %b, i64 %c)
  ret void
}
declare void @llvm.ppc.vsx.stxvrll.v2i64(<2 x i64>, ptr, i64)

; Test for load/store vectore pair.

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

define void @testSTXVPRL(ptr %v, ptr %vp, i64 %len) {
; CHECK-LABEL: testSTXVPRL:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2
; CHECK-NEXT:    lxv v3
; CHECK-NEXT:    stxvprl vsp34, r4, r5
; CHECK-NEXT:    blr
entry:
  %0 = load <256 x i1>, ptr %v, align 32
  tail call void @llvm.ppc.vsx.stxvprl(<256 x i1> %0, ptr %vp, i64 %len)
  ret void
}
declare void @llvm.ppc.vsx.stxvprl(<256 x i1>, ptr, i64)

define void @testSTXVPRLL(ptr %v, ptr %vp, i64 %len) {
; CHECK-LABEL: testSTXVPRLL:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2
; CHECK-NEXT:    lxv v3
; CHECK-NEXT:    stxvprll vsp34, r4, r5
; CHECK-NEXT:    blr
entry:
  %0 = load <256 x i1>, ptr %v, align 32
  tail call void @llvm.ppc.vsx.stxvprll(<256 x i1> %0, ptr %vp, i64 %len)
  ret void
}
declare void @llvm.ppc.vsx.stxvprll(<256 x i1>, ptr, i64)
