; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i8 0, align 1

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igeuc(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp uge i8 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
; CHECK-LABEL: test_igeuc:
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK-NEXT: not [[REG2:r[0-9]+]], [[REG1]]
; CHECK-NEXT: rldicl r3, [[REG2]], 1, 63
; CHECK: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igeuc_sext(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp uge i8 %a, %b
  %sub = sext i1 %cmp to i32
  ret i32 %sub
; CHECK-LABEL: @test_igeuc_sext
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK-NEXT: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-NEXT: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK-NEXT: blr  
  
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igeuc_z(i8 zeroext %a) {
entry:
  %cmp = icmp uge i8 %a, 0
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
; CHECK-LABEL: @test_igeuc_z
; CHECK: li r3, 1
; CHECK-NEXT: blr  
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igeuc_sext_z(i8 zeroext %a) {
entry:
  %cmp = icmp uge i8 %a, 0
  %conv2 = sext i1 %cmp to i32
  ret i32 %conv2
; CHECK-LABEL: @test_igeuc_sext_z
; CHECK: li r3, -1
; CHECK-NEXT: blr  
}

; Function Attrs: norecurse nounwind
define void @test_igeuc_store(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp uge i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @glob
  ret void
; CHECK_LABEL: test_igeuc_store:
; CHECK: sub [[REG1:r[0-9]+]], r3, r4
; CHECK: not [[REG2:r[0-9]+]], [[REG1]]
; CHECK: rldicl r3, [[REG2]], 1, 63
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @test_igeuc_sext_store(i8 zeroext %a, i8 zeroext %b) {
entry:
  %cmp = icmp uge i8 %a, %b
  %conv3 = sext i1 %cmp to i8
  store i8 %conv3, i8* @glob
  ret void
; CHECK-TBD-LABEL: @test_igeuc_sext_store
; CHECK-TBD: subf [[REG1:r[0-9]+]], r3, r4
; CHECK-TBD: rldicl [[REG2:r[0-9]+]], [[REG1]], 1, 63
; CHECK-TBD: addi [[REG3:r[0-9]+]], [[REG2]], -1
; CHECK-TBD: stb  [[REG3]]
; CHECK-TBD: blr    
}

; Function Attrs : norecurse nounwind
define void @test_igeuc_z_store(i8 zeroext %a) {
entry:
  %cmp = icmp uge i8 %a, 0
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @glob
  ret void
; CHECK-LABEL: @test_igeuc_z_store
; CHECK: li [[REG1:r[0-9]+]], 1
; CHECK: stb [[REG1]]
; CHECK: blr    
}

; Function Attrs: norecurse nounwind
define void @test_igeuc_sext_z_store(i8 zeroext %a) {
entry:
  %cmp = icmp uge i8 %a, 0
  %conv3 = sext i1 %cmp to i8
  store i8 %conv3, i8* @glob
  ret void
; CHECK-LABEL: @test_igeuc_sext_z_store
; CHECK: li [[REG1:r[0-9]+]], -1
; CHECK: stb [[REG1]]
; CHECK: blr
}
