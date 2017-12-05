; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i64 0, align 8

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtsll(i64 %a, i64 %b) {
; CHECK-LABEL: test_igtsll:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    sradi [[REG1:r[0-9]+]], r4, 63
; CHECK-NEXT:    rldicl [[REG2:r[0-9]+]], r3, 1, 63
; CHECK-NEXT:    subfc [[REG3:r[0-9]+]], r3, r4
; CHECK-NEXT:    adde [[REG4:r[0-9]+]], [[REG2]], [[REG1]]
; CHECK-NEXT:    xori r3, [[REG4]], 1
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i64 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtsll_sext(i64 %a, i64 %b) {
; CHECK-LABEL: test_igtsll_sext:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    sradi [[REG1:r[0-9]+]], r4, 63
; CHECK-NEXT:    rldicl [[REG2:r[0-9]+]], r3, 1, 63
; CHECK-NEXT:    subfc [[REG3:r[0-9]+]], r3, r4
; CHECK-NEXT:    adde [[REG4:r[0-9]+]], [[REG2]], [[REG1]]
; CHECK-NEXT:    xori [[REG5:r[0-9]+]], [[REG4]], 1
; CHECK-NEXT:    neg r3, [[REG5]]
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i64 %a, %b
  %sub = sext i1 %cmp to i32
  ret i32 %sub
}

; FIXME
; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtsll_z(i64 %a) {
; CHECK-LABEL: test_igtsll_z:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addi r4, r3, -1
; CHECK-NEXT:    nor r3, r4, r3
; CHECK-NEXT:    rldicl r3, r3, 1, 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i64 %a, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @test_igtsll_sext_z(i64 %a) {
; CHECK-LABEL: test_igtsll_sext_z:
; CHECK:    addi [[REG1:r[0-9]+]], r3, -1
; CHECK-NEXT:    nor [[REG2:r[0-9]+]], [[REG1]], r3
; CHECK-NEXT:    sradi r3, [[REG2]], 63
entry:
  %cmp = icmp sgt i64 %a, 0
  %sub = sext i1 %cmp to i32
  ret i32 %sub
}

; Function Attrs: norecurse nounwind
define void @test_igtsll_store(i64 %a, i64 %b) {
; CHECK-LABEL: test_igtsll_store:
; CHECK:       # %bb.0: # %entry
; CHECK:         sradi [[REG1:r[0-9]+]], r4, 63
; CHECK:         rldicl [[REG2:r[0-9]+]], r3, 1, 63
; CHECK-DIAG:    subfc [[REG3:r[0-9]+]], r3, r4
; CHECK:         adde [[REG4:r[0-9]+]], [[REG2]], [[REG1]]
; CHECK:         xori [[REG5:r[0-9]+]], [[REG4]], 1
; CHECK-NOT:     neg
entry:
  %cmp = icmp sgt i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @glob, align 8
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_igtsll_sext_store(i64 %a, i64 %b) {
; CHECK-LABEL: test_igtsll_sext_store:
; CHECK:       # %bb.0: # %entry
; CHECK:         sradi [[REG1:r[0-9]+]], r4, 63
; CHECK:         rldicl [[REG2:r[0-9]+]], r3, 1, 63
; CHECK-DIAG:    subfc [[REG3:r[0-9]+]], r3, r4
; CHECK:         adde [[REG4:r[0-9]+]], [[REG2]], [[REG1]]
; CHECK:         xori [[REG5:r[0-9]+]], [[REG4]], 1
; CHECK:         neg {{r[0-9]+}}, [[REG5]]
entry:
  %cmp = icmp sgt i64 %a, %b
  %conv1 = sext i1 %cmp to i64
  store i64 %conv1, i64* @glob, align 8
  ret void
}

; FIXME
; Function Attrs: norecurse nounwind
define void @test_igtsll_z_store(i64 %a) {
; CHECK-LABEL: test_igtsll_z_store:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-NEXT:    addi r5, r3, -1
; CHECK-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-NEXT:    nor r3, r5, r3
; CHECK-NEXT:    rldicl r3, r3, 1, 63
; CHECK-NEXT:    std r3, 0(r4)
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i64 %a, 0
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @glob, align 8
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_igtsll_sext_z_store(i64 %a) {
; CHECK-LABEL: test_igtsll_sext_z_store:
; CHECK:    addi [[REG1:r[0-9]+]], r3, -1
; CHECK:    nor [[REG2:r[0-9]+]], [[REG1]], r3
; CHECK:    sradi [[REG3:r[0-9]+]], [[REG2]], 63
entry:
  %cmp = icmp sgt i64 %a, 0
  %conv1 = sext i1 %cmp to i64
  store i64 %conv1, i64* @glob, align 8
  ret void
}
