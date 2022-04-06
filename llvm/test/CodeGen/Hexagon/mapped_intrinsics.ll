; RUN: llc -march=hexagon -debug-only=isel < %s 2>&1 | FileCheck %s
; REQUIRES: asserts, abi_breaking_checks

; This test validates that ISel picks the correct equivalent of below mentioned intrinsics
; For S2_asr_i_r_rnd_goodsyntax:
;   if (#u5 == 0) Assembler mapped to: Rd = Rs
;   else Rd = asr(Rs,#u5-1):rnd
; For S2_asr_i_p_rnd_goodsyntax:
;   if (#u6 == 0) Assembler mapped to: Rdd = combine(Rss.H32,Rss.L32)
;   else Rdd = asr(Rss,#u6-1):rnd
; For S5_vasrhrnd_goodsyntax:
;   if (#u4 == 0) Assembler mapped to: Rdd = combine(Rss.H32,Rss.L32)
;   else Rdd = vasrh(Rss,#u4-1):raw
; For S5_asrhub_rnd_sat_goodsyntax:
;   if (#u4 == 0) Assembler mapped to: Rd = vsathub(Rss)
;   else Rd = vasrhub(Rss,#u4-1):raw

target triple = "hexagon-unknown--elf"

; CHECK-LABEL: f0
; CHECK: ISEL: Starting selection on{{.*}}llvm.hexagon.S2.asr.i.r.rnd.goodsyntax
; CHECK: Morphed node{{.*}}A2_tfr
define i32 @f0(i32 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = tail call i32 @llvm.hexagon.S2.asr.i.r.rnd.goodsyntax(i32 %a0, i32 0)
  %v1 = add i32 %v0, %a1
  ret i32 %v1
}

declare i32 @llvm.hexagon.S2.asr.i.r.rnd.goodsyntax(i32, i32) #1

; CHECK-LABEL: f1
; CHECK: ISEL: Starting selection on{{.*}}llvm.hexagon.S2.asr.i.r.rnd.goodsyntax
; CHECK: Morphed node{{.*}}S2_asr_i_r_rnd
define i32 @f1(i32 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = tail call i32 @llvm.hexagon.S2.asr.i.r.rnd.goodsyntax(i32 %a0, i32 9)
  %v1 = add i32 %v0, %a1
  ret i32 %v1
}

; CHECK-LABEL: f2
; CHECK: ISEL: Starting selection on{{.*}}llvm.hexagon.S2.asr.i.p.rnd.goodsyntax
; CHECK: Morphed node{{.*}}A2_combinew
define i64 @f2(i64 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = zext i32 %a1 to i64
  %v1 = tail call i64 @llvm.hexagon.S2.asr.i.p.rnd.goodsyntax(i64 %a0, i32 0)
  %v2 = add nsw i64 %v1, %v0
  ret i64 %v2
}

declare i64 @llvm.hexagon.S2.asr.i.p.rnd.goodsyntax(i64, i32) #1

; CHECK-LABEL: f3
; CHECK: ISEL: Starting selection on{{.*}}llvm.hexagon.S2.asr.i.p.rnd.goodsyntax
; CHECK: Morphed node{{.*}}S2_asr_i_p_rnd
define i64 @f3(i64 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = zext i32 %a1 to i64
  %v1 = tail call i64 @llvm.hexagon.S2.asr.i.p.rnd.goodsyntax(i64 %a0, i32 9)
  %v2 = add nsw i64 %v1, %v0
  ret i64 %v2
}

; CHECK-LABEL: f4
; CHECK: ISEL: Starting selection on{{.*}}llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax
; CHECK: Morphed node{{.*}}S2_vsathub
define i32 @f4(i64 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = tail call i32 @llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax(i64 %a0, i32 0)
  %v1 = add i32 %v0, %a1
  ret i32 %v1
}

declare i32 @llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax(i64, i32) #1

; CHECK-LABEL: f5
; CHECK: ISEL: Starting selection on{{.*}}llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax
; CHECK: Morphed node{{.*}}S5_asrhub_rnd_sat
define i32 @f5(i64 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = tail call i32 @llvm.hexagon.S5.asrhub.rnd.sat.goodsyntax(i64 %a0, i32 9)
  %v1 = add i32 %v0, %a1
  ret i32 %v1
}

; CHECK-LABEL: f6
; CHECK: ISEL: Starting selection on{{.*}}llvm.hexagon.S5.vasrhrnd.goodsyntax
; CHECK: Morphed node{{.*}}A2_combinew
define i64 @f6(i64 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = zext i32 %a1 to i64
  %v1 = tail call i64 @llvm.hexagon.S5.vasrhrnd.goodsyntax(i64 %a0, i32 0)
  %v2 = add nsw i64 %v1, %v0
  ret i64 %v2
}

declare i64 @llvm.hexagon.S5.vasrhrnd.goodsyntax(i64, i32) #1

; CHECK-LABEL: f7
; CHECK: ISEL: Starting selection on{{.*}}llvm.hexagon.S5.vasrhrnd.goodsyntax
; CHECK: Morphed node{{.*}}S5_vasrhrnd
define i64 @f7(i64 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = zext i32 %a1 to i64
  %v1 = tail call i64 @llvm.hexagon.S5.vasrhrnd.goodsyntax(i64 %a0, i32 9)
  %v2 = add nsw i64 %v1, %v0
  ret i64 %v2
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
