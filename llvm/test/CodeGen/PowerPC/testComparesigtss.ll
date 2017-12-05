; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:   --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:   --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i16 0, align 2

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtss(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: test_igtss:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    sub [[REG1:r[0-9]+]], r4, r3
; CHECK-NEXT:    rldicl r3, [[REG1]], 1, 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtss_sext(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: test_igtss_sext:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    sub [[REG:r[0-9]+]], r4, r3
; CHECK-NEXT:    sradi r3, [[REG]], 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i16 %a, %b
  %sub = sext i1 %cmp to i32
  ret i32 %sub
}

; FIXME
; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtss_z(i16 signext %a) {
; CHECK-LABEL: test_igtss_z:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    neg r3, r3
; CHECK-NEXT:    rldicl r3, r3, 1, 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i16 %a, 0
  %conv1 = zext i1 %cmp to i32
  ret i32 %conv1
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtss_sext_z(i16 signext %a) {
; CHECK-LABEL: test_igtss_sext_z:
; CHECK:       # %bb.0: # %entry
; CHECK:    neg [[REG2:r[0-9]+]], r3
; CHECK-NEXT:    sradi r3, [[REG2]], 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i16 %a, 0
  %sub = sext i1 %cmp to i32
  ret i32 %sub
}

; Function Attrs: norecurse nounwind
define void @test_igtss_store(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: test_igtss_store:
; CHECK:       # %bb.0: # %entry
; CHECK:         sub [[REG1:r[0-9]+]], r4, r3
; CHECK:         rldicl {{r[0-9]+}}, [[REG1]], 1, 63
entry:
  %cmp = icmp sgt i16 %a, %b
  %conv3 = zext i1 %cmp to i16
  store i16 %conv3, i16* @glob, align 2
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_igtss_sext_store(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: test_igtss_sext_store:
; CHECK:       # %bb.0: # %entry
; CHECK:         sub [[REG:r[0-9]+]], r4, r3
; CHECK:         sradi {{r[0-9]+}}, [[REG]], 63
entry:
  %cmp = icmp sgt i16 %a, %b
  %conv3 = sext i1 %cmp to i16
  store i16 %conv3, i16* @glob, align 2
  ret void
}

; FIXME
; Function Attrs: norecurse nounwind
define void @test_igtss_z_store(i16 signext %a) {
; CHECK-LABEL: test_igtss_z_store:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-NEXT:    neg r3, r3
; CHECK-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-NEXT:    rldicl r3, r3, 1, 63
; CHECK-NEXT:    sth r3, 0(r4)
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i16 %a, 0
  %conv2 = zext i1 %cmp to i16
  store i16 %conv2, i16* @glob, align 2
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_igtss_sext_z_store(i16 signext %a) {
; CHECK-LABEL: test_igtss_sext_z_store:
; CHECK:       neg [[REG2:r[0-9]+]], r3
; CHECK:       sradi {{r[0-9]+}}, [[REG2]], 63
entry:
  %cmp = icmp sgt i16 %a, 0
  %conv2 = sext i1 %cmp to i16
  store i16 %conv2, i16* @glob, align 2
  ret void
}
