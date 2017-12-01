; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igeui(i32 zeroext %a, i32 zeroext %b) {
entry:
  %cmp = icmp uge i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: test_igeui:
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG2]], 1, 63
; CHECK-NEXT: xori r3, [[REG2]], 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igeui_sext(i32 zeroext %a, i32 zeroext %b) {
entry:
  %cmp = icmp uge i32 %a, %b
  %sub = sext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: @test_igeui_sext
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK-NEXT: blr    
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igeui_z(i32 zeroext %a) {
entry:
  %cmp = icmp uge i32 %a, 0
  %sub = zext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: @test_igeui_z
; CHECK: li r3, 1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igeui_sext_z(i32 zeroext %a) {
entry:
  %cmp = icmp uge i32 %a, 0
  %sub = sext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: @test_igeui_sext_z
; CHECK: li r3, -1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @test_igeui_store(i32 zeroext %a, i32 zeroext %b) {
entry:
  %cmp = icmp uge i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @glob
  ret void
; CHECK_LABEL: test_igeuc_store:
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG2]], 1, 63
; CHECK: xori {{r[0-9]+}}, [[REG2]], 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_igeui_sext_store(i32 zeroext %a, i32 zeroext %b) {
entry:
  %cmp = icmp uge i32 %a, %b
  %sub = sext i1 %cmp to i32
  store i32 %sub, i32* @glob
  ret void
; CHECK-LABEL: @test_igeui_sext_store
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK: stw  [[REG3]]
; CHECK: blr    
}

; Function Attrs: norecurse nounwind
define void @test_igeui_z_store(i32 zeroext %a) {
entry:
  %cmp = icmp uge i32 %a, 0
  %conv1 = zext i1 %cmp to i32
  store i32 %conv1, i32* @glob
  ret void
; CHECK-LABEL: @test_igeui_z_store
; CHECK: li [[REG1:r[0-9]+]], 1
; CHECK: stw [[REG1]]
; CHECK: blr  
}

; Function Attrs: norecurse nounwind
define void @test_igeui_sext_z_store(i32 zeroext %a) {
entry:
  %cmp = icmp uge i32 %a, 0 
  %conv1 = sext i1 %cmp to i32
  store i32 %conv1, i32* @glob
  ret void
; CHECK-LABEL: @test_igeui_sext_z_store
; CHECK: li [[REG1:r[0-9]+]], 0
; CHECK: oris [[REG2:r[0-9]+]], [[REG1]], 65535
; CHECK: ori [[REG3:r[0-9]+]], [[REG2]], 65535
; CHECK: stw [[REG3]]
; CHECK: blr
}

