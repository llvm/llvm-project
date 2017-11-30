; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i8 0, align 1

; Function Attrs: norecurse nounwind readnone
define i64 @test_llleuc(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp ule i8 %a, %b
  %conv3 = zext i1 %cmp to i64
  ret i64 %conv3
; CHECK-LABEL: test_llleuc:
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: xori r3, [[REG2]], 1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llleuc_sext(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp ule i8 %a, %b
  %conv3 = sext i1 %cmp to i64
  ret i64 %conv3
; CHECK-LABEL: @test_llleuc_sext
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llleuc_z(i8 zeroext %a) {
entry:
  %cmp = icmp ule i8 %a, 0
  %conv2 = zext i1 %cmp to i64
  ret i64 %conv2
; CHECK-LABEL: test_llleuc_z:
; CHECK: cntlzw r3, r3
; CHECK-NEXT: srwi r3, r3, 5
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llleuc_sext_z(i8 zeroext %a) {
entry:
  %cmp = icmp ule i8 %a, 0
  %conv2 = sext i1 %cmp to i64
  ret i64 %conv2
; CHECK-LABEL: @test_llleuc_sext_z
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK-NEXT: srwi [[REG2:r[0-9]+]], [[REG1]], 5
; CHECK-NEXT: neg r3, [[REG2]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @test_llleuc_store(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp ule i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @glob
  ret void
; CHECK-LABEL: test_llleuc_store:
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK: xori {{r[0-9]+}}, [[REG2]], 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_llleuc_sext_store(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp ule i8 %a, %b
  %conv3 = sext i1 %cmp to i8
  store i8 %conv3, i8* @glob
  ret void
; CHECK-LABEL: @test_llleuc_sext_store
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK: stb  [[REG3]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_llleuc_z_store(i8 zeroext %a) {
entry:
  %cmp = icmp ule i8 %a, 0
  %conv2 = zext i1 %cmp to i8
  store i8 %conv2, i8* @glob
  ret void
; CHECK-LABEL: test_llleuc_z_store:
; CHECK: cntlzw r3, r3
; CHECK: srwi {{r[0-9]}}, r3, 5
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_llleuc_sext_z_store(i8 zeroext %a) {
entry:
  %cmp = icmp ule i8 %a, 0
  %conv2 = sext i1 %cmp to i8
  store i8 %conv2, i8* @glob
  ret void
; CHECK-LABEL: @test_llleuc_sext_z_store
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK: srwi [[REG2:r[0-9]+]], [[REG1]], 5
; CHECK: neg [[REG3:r[0-9]+]], [[REG2]]
; CHECK: stb [[REG3]]
; CHECK: blr
}
