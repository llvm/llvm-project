; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i8 0, align 1

; Function Attrs: norecurse nounwind readnone
define i64 @test_llgtuc(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: test_llgtuc:
; CHECK:    sub [[REG:r[0-9]+]], r4, r3
; CHECK:    rldicl r3, [[REG]], 1, 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp ugt i8 %a, %b
  %conv3 = zext i1 %cmp to i64
  ret i64 %conv3
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llgtuc_sext(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: test_llgtuc_sext:
; CHECK:    sub [[REG:r[0-9]+]], r4, r3
; CHECK:    sradi r3, [[REG]], 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp ugt i8 %a, %b
  %conv3 = sext i1 %cmp to i64
  ret i64 %conv3
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llgtuc_z(i8 zeroext %a) {
; CHECK-LABEL: test_llgtuc_z:
; CHECK:    cntlzw r3, r3
; CHECK:    srwi r3, r3, 5
; CHECK:    xori r3, r3, 1
; CHECK-NEXT:    blr
entry:
  %cmp = icmp ne i8 %a, 0
  %conv2 = zext i1 %cmp to i64
  ret i64 %conv2
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llgtuc_sext_z(i8 zeroext %a) {
; CHECK-LABEL: test_llgtuc_sext_z:
; CHECK:    cntlzw r3, r3
; CHECK:    srwi r3, r3, 5
; CHECK:    xori r3, r3, 1
; CHECK:    neg r3, r3
; CHECK-NEXT:    blr
entry:
  %cmp = icmp ne i8 %a, 0
  %conv2 = sext i1 %cmp to i64
  ret i64 %conv2
}

; Function Attrs: norecurse nounwind
define void @test_llgtuc_store(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: test_llgtuc_store:
; CHECK:         sub [[REG:r[0-9]+]], r4, r3
; CHECK:         rldicl {{r[0-9]+}}, [[REG]], 1, 63
entry:
  %cmp = icmp ugt i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @glob, align 1
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_llgtuc_sext_store(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: test_llgtuc_sext_store:
; CHECK:         sub [[REG:r[0-9]+]], r4, r3
; CHECK:         sradi {{r[0-9]+}}, [[REG]], 63
entry:
  %cmp = icmp ugt i8 %a, %b
  %conv3 = sext i1 %cmp to i8
  store i8 %conv3, i8* @glob, align 1
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_llgtuc_z_store(i8 zeroext %a) {
; CHECK-LABEL: test_llgtuc_z_store:
; CHECK:    cntlzw r3, r3
; CHECK:    srwi r3, r3, 5
; CHECK:    xori r3, r3, 1
; CHECK:    stb r3, 0(r4)
; CHECK-NEXT:    blr
entry:
  %cmp = icmp ne i8 %a, 0
  %conv2 = zext i1 %cmp to i8
  store i8 %conv2, i8* @glob, align 1
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_llgtuc_sext_z_store(i8 zeroext %a) {
; CHECK-LABEL: test_llgtuc_sext_z_store:
; CHECK:    cntlzw r3, r3
; CHECK:    srwi r3, r3, 5
; CHECK:    xori r3, r3, 1
; CHECK:    neg r3, r3
; CHECK:    stb r3, 0(r4)
; CHECK-NEXT:    blr
entry:
  %cmp = icmp ne i8 %a, 0
  %conv2 = sext i1 %cmp to i8
  store i8 %conv2, i8* @glob, align 1
  ret void
}
