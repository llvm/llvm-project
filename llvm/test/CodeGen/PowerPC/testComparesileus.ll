; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i16 0, align 2

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_ileus(i16 zeroext %a, i16 zeroext %b) {
entry:
  %cmp = icmp ule i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
; CHECK-LABEL: test_ileus:
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: xori r3, [[REG2]], 1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_ileus_sext(i16 zeroext %a, i16 zeroext %b) {
entry:
  %cmp = icmp ule i16 %a, %b
  %sub = sext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: @test_ileus_sext
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_ileus_z(i16 zeroext %a) {
entry:
  %cmp = icmp ule i16 %a, 0
  %conv1 = zext i1 %cmp to i32
  ret i32 %conv1
; CHECK-LABEL: test_ileus_z:
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK: srwi r3, [[REG1]], 5
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_ileus_sext_z(i16 zeroext %a) {
entry:
  %cmp = icmp ule i16 %a, 0
  %sub = sext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: @test_ileus_sext_z
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK-NEXT: srwi [[REG2:r[0-9]+]], [[REG1]], 5
; CHECK-NEXT: neg r3, [[REG2]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @test_ileus_store(i16 zeroext %a, i16 zeroext %b) {
entry:
  %cmp = icmp ule i16 %a, %b
  %conv3 = zext i1 %cmp to i16
  store i16 %conv3, i16* @glob
  ret void
; CHECK-LABEL: test_ileus_store:
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK: xori {{r[0-9]+}}, [[REG2]], 1
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_ileus_sext_store(i16 zeroext %a, i16 zeroext %b) {
entry:
  %cmp = icmp ule i16 %a, %b
  %conv3 = sext i1 %cmp to i16
  store i16 %conv3, i16* @glob
  ret void
; CHECK-LABEL: @test_ileus_sext_store
; CHECK: sub [[REG1:r[0-9]+]], r4, r3
; CHECK: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK: sth  [[REG3]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_ileus_z_store(i16 zeroext %a) {
entry:
  %cmp = icmp ule i16 %a, 0
  %conv2 = zext i1 %cmp to i16
  store i16 %conv2, i16* @glob
  ret void
; CHECK-LABEL: test_ileus_z_store:
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK: srwi {{r[0-9]+}}, [[REG1]], 5
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_ileus_sext_z_store(i16 zeroext %a) {
entry:
  %cmp = icmp ule i16 %a, 0
  %conv2 = sext i1 %cmp to i16
  store i16 %conv2, i16* @glob
  ret void
; CHECK-LABEL: @test_ileus_sext_z_store
; CHECK: cntlzw [[REG1:r[0-9]+]], r3
; CHECK: srwi [[REG2:r[0-9]+]], [[REG1]], 5
; CHECK: neg [[REG3:r[0-9]+]], [[REG2]]
; CHECK: sth [[REG3]]
; CHECK: blr
}

