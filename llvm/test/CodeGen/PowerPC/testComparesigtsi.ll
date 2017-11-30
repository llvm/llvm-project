; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtsi(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: test_igtsi:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    sub [[REG:r[0-9]+]], r4, r3
; CHECK-NEXT:    rldicl r3, [[REG]], 1, 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtsi_sext(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: test_igtsi_sext:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    sub [[REG:r[0-9]+]], r4, r3
; CHECK-NEXT:    sradi r3, [[REG]], 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i32 %a, %b
  %sub = sext i1 %cmp to i32
  ret i32 %sub
}

; FIXME
; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtsi_z(i32 signext %a) {
; CHECK-LABEL: test_igtsi_z:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    neg r3, r3
; CHECK-NEXT:    rldicl r3, r3, 1, 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i32 %a, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtsi_sext_z(i32 signext %a) {
; CHECK-LABEL: test_igtsi_sext_z:
; CHECK:  neg [[REG2:r[0-9]+]], r3
; CHECK-NEXT: sradi r3, [[REG2]], 63
; CHECK-NEXT: blr
entry:
  %cmp = icmp sgt i32 %a, 0
  %sub = sext i1 %cmp to i32
  ret i32 %sub
}

; Function Attrs: norecurse nounwind
define void @test_igtsi_store(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: test_igtsi_store:
; CHECK:       # BB#0: # %entry
; CHECK:         sub [[REG:r[0-9]+]], r4, r3
; CHECK:         rldicl {{r[0-9]+}}, [[REG]], 1, 63
entry:
  %cmp = icmp sgt i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @glob, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_igtsi_sext_store(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: test_igtsi_sext_store:
; CHECK:       # BB#0: # %entry
; CHECK:         sub [[REG:r[0-9]+]], r4, r3
; CHECK:         sradi {{r[0-9]+}}, [[REG]], 63
entry:
  %cmp = icmp sgt i32 %a, %b
  %sub = sext i1 %cmp to i32
  store i32 %sub, i32* @glob, align 4
  ret void
}

; FIXME
; Function Attrs: norecurse nounwind
define void @test_igtsi_z_store(i32 signext %a) {
; CHECK-LABEL: test_igtsi_z_store:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-NEXT:    neg r3, r3
; CHECK-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-NEXT:    rldicl r3, r3, 1, 63
; CHECK-NEXT:    stw r3, 0(r4)
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i32 %a, 0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @glob, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_igtsi_sext_z_store(i32 signext %a) {
; CHECK-LABEL: test_igtsi_sext_z_store:
; CHECK: neg [[REG:r[0-9]+]], r3
; CHECK: sradi {{r[0-9]+}}, [[REG]], 63
entry:
  %cmp = icmp sgt i32 %a, 0
  %sub = sext i1 %cmp to i32
  store i32 %sub, i32* @glob, align 4
  ret void
}
