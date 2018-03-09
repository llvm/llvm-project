; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_ileui(i32 zeroext %a, i32 zeroext %b) {
entry:
  %cmp = icmp ule i32 %a, %b
  %sub = zext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: test_ileui:
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: xori r3, [[REG2]], 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_ileui_sext(i32 zeroext %a, i32 zeroext %b) {
entry:
  %cmp = icmp ule i32 %a, %b
  %sub = sext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: @test_ileui_sext
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_ileui_z(i32 zeroext %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  %sub = zext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: test_ileui_z:
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK: srwi r3, [[REG1]], 5
; CHECK: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_ileui_sext_z(i32 zeroext %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  %sub = sext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: @test_ileui_sext_z
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK-NEXT: srwi [[REG2:r[0-9]+]], [[REG1]], 5
; CHECK-NEXT: neg r3, [[REG2]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_ileui_store(i32 zeroext %a, i32 zeroext %b) {
entry:
  %cmp = icmp ule i32 %a, %b
  %sub = zext i1 %cmp to i32
  store i32 %sub, i32* @glob
  ret void
; CHECK-LABEL: test_ileui_store:
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK: xori {{r[0-9]+}}, [[REG2]], 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_ileui_sext_store(i32 zeroext %a, i32 zeroext %b) {
entry:
  %cmp = icmp ule i32 %a, %b
  %sub = sext i1 %cmp to i32
  store i32 %sub, i32* @glob
  ret void
; CHECK-LABEL: @test_ileui_sext_store
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK: stw  [[REG3]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_ileui_z_store(i32 zeroext %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  %sub = zext i1 %cmp to i32
  store i32 %sub, i32* @glob
  ret void
; CHECK-LABEL: test_ileui_z_store:
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK: srwi {{r[0-9]+}}, [[REG1]], 5
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_ileui_sext_z_store(i32 zeroext %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  %sub = sext i1 %cmp to i32
  store i32 %sub, i32* @glob
  ret void
; CHECK-LABEL: @test_ileui_sext_z_store
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK: srwi [[REG2:r[0-9]+]], [[REG1]], 5
; CHECK: neg [[REG3:r[0-9]+]], [[REG2]]
; CHECK: stw [[REG3]]
; CHECK: blr
}

