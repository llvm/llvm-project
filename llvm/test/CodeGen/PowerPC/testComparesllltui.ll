; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

@glob = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readnone
define i64 @test_llltui(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: test_llltui:
; CHECK:       # %bb.0: # %entry
; CHECK-NOT:     clrldi
; CHECK-NEXT:    sub [[REG:r[0-9]+]], r3, r4
; CHECK-NEXT:    rldicl r3, [[REG]], 1, 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp ult i32 %a, %b
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llltui_sext(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: test_llltui_sext:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    sub [[REG:r[0-9]+]], r3, r4
; CHECK-NEXT:    sradi r3, [[REG]], 63
; CHECK-NEXT:    blr
entry:
  %cmp = icmp ult i32 %a, %b
  %conv1 = sext i1 %cmp to i64
  ret i64 %conv1
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llltui_z(i32 zeroext %a) {
; CHECK-LABEL: test_llltui_z:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    li r3, 0
; CHECK-NEXT:    blr
entry:
  ret i64 0
}

; Function Attrs: norecurse nounwind readnone
define i64 @test_llltui_sext_z(i32 zeroext %a) {
; CHECK-LABEL: test_llltui_sext_z:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    li r3, 0
; CHECK-NEXT:    blr
entry:
  ret i64 0
}

; Function Attrs: norecurse nounwind
define void @test_llltui_store(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: test_llltui_store:
; CHECK:       # %bb.0: # %entry
; CHECK-NOT:     clrldi
; CHECK:         sub [[REG:r[2-9]+]], r3, r4
; CHECK:         rldicl {{r[0-9]+}}, [[REG]], 1, 63
entry:
  %cmp = icmp ult i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @glob, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_llltui_sext_store(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: test_llltui_sext_store:
; CHECK:       # %bb.0: # %entry
; CHECK-NOT:     clrldi
; CHECK:         sub [[REG:r[0-9]+]], r3, r4
; CHECK:         sradi {{r[0-9]+}}, [[REG]], 63
entry:
  %cmp = icmp ult i32 %a, %b
  %sub = sext i1 %cmp to i32
  store i32 %sub, i32* @glob, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_llltui_z_store(i32 zeroext %a) {
; CHECK-LABEL: test_llltui_z_store:
; CHECK:       # %bb.0: # %entry
; CHECK:         li [[REG:r[0-9]+]], 0
; CHECK:         stw [[REG]], 0(r3)
; CHECK-NEXT:    blr
entry:
  store i32 0, i32* @glob, align 4
  ret void
}

; Function Attrs: norecurse nounwind
define void @test_llltui_sext_z_store(i32 zeroext %a) {
; CHECK-LABEL: test_llltui_sext_z_store:
; CHECK:       # %bb.0: # %entry
; CHECK:         li [[REG:r[0-9]+]], 0
; CHECK:         stw [[REG]], 0(r3)
; CHECK-NEXT:    blr
entry:
  store i32 0, i32* @glob, align 4
  ret void
}

