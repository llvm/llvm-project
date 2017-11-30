; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i8 0, align 1

; Function Attrs: norecurse nounwind readnone
define i64 @test_llgeuc(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp uge i8 %a, %b
  %conv3 = zext i1 %cmp to i64
  ret i64 %conv3
; CHECK-LABEL: test_llgeuc:
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG2]], 1, 63
; CHECK: xori r3, [[REG2]], 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llgeuc_sext(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp uge i8 %a, %b
  %conv3 = sext i1 %cmp to i64
  ret i64 %conv3
; CHECK-LABEL: @test_llgeuc_sext
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llgeuc_z(i8 zeroext %a) {
entry:
  %cmp = icmp uge i8 %a, 0
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
; CHECK-LABEL: @test_llgeuc_z
; CHECK: li r3, 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llgeuc_sext_z(i8 zeroext %a) {
entry:
  %cmp = icmp uge i8 %a, 0
  %conv1 = sext i1 %cmp to i64
  ret i64 %conv1
; CHECK-LABEL: @test_llgeuc_sext_z
; CHECK: li r3, -1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @test_llgeuc_store(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp uge i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @glob
  ret void
; CHECK_LABEL: test_llgeuc_store:
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG2]], 1, 63
; CHECK: xori {{r[0-9]+}}, [[REG2]], 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_llgeuc_sext_store(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp uge i8 %a, %b
  %conv3 = sext i1 %cmp to i8
  store i8 %conv3, i8* @glob
  ret void
; CHECK-LABEL: @test_llgeuc_sext_store
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK: stb  [[REG3]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_llgeuc_z_store(i8 zeroext %a) {
entry:
  %cmp = icmp uge i8 %a, 0
  %conv1 = zext i1 %cmp to i8
  store i8 %conv1, i8* @glob
  ret void
; CHECK-LABEL: @test_llgeuc_z_store
; CHECK: li [[REG1:r[0-9]+]], 1
; CHECK: stb [[REG1]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_llgeuc_sext_z_store(i8 zeroext %a) {
entry:
  %cmp = icmp uge i8 %a, 0
  %conv1 = sext i1 %cmp to i8
  store i8 %conv1, i8* @glob
  ret void
; CHECK-LABEL: @test_llgeuc_sext_z_store
; CHECK: li [[REG1:r[0-9]+]], 255
; CHECK: stb [[REG1]]
; CHECK: blr
}

