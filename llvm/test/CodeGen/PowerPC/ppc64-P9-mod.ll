; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s -check-prefix=CHECK-PWR8 -implicit-check-not mod[us][wd]

@mod_resultsw = local_unnamed_addr global i32 0, align 4
@mod_resultud = local_unnamed_addr global i64 0, align 8
@div_resultsw = local_unnamed_addr global i32 0, align 4
@mod_resultuw = local_unnamed_addr global i32 0, align 4
@div_resultuw = local_unnamed_addr global i32 0, align 4
@div_resultsd = local_unnamed_addr global i64 0, align 8
@mod_resultsd = local_unnamed_addr global i64 0, align 8
@div_resultud = local_unnamed_addr global i64 0, align 8

; Function Attrs: norecurse nounwind
define void @modulo_sw(i32 signext %a, i32 signext %b) local_unnamed_addr {
entry:
  %rem = srem i32 %a, %b
  store i32 %rem, i32* @mod_resultsw, align 4
  ret void
; CHECK-LABEL: modulo_sw
; CHECK: modsw {{[0-9]+}}, 3, 4
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_sw
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @modulo_uw(i32 zeroext %a, i32 zeroext %b) local_unnamed_addr {
entry:
  %rem = urem i32 %a, %b
  ret i32 %rem
; CHECK-LABEL: modulo_uw
; CHECK: moduw {{[0-9]+}}, 3, 4
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_uw
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @modulo_sd(i64 %a, i64 %b) local_unnamed_addr {
entry:
  %rem = srem i64 %a, %b
  ret i64 %rem
; CHECK-LABEL: modulo_sd
; CHECK: modsd {{[0-9]+}}, 3, 4
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_sd
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind
define void @modulo_ud(i64 %a, i64 %b) local_unnamed_addr {
entry:
  %rem = urem i64 %a, %b
  store i64 %rem, i64* @mod_resultud, align 8
  ret void
; CHECK-LABEL: modulo_ud
; CHECK: modud {{[0-9]+}}, 3, 4
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_ud
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind
define void @modulo_div_sw(i32 signext %a, i32 signext %b) local_unnamed_addr {
entry:
  %rem = srem i32 %a, %b
  store i32 %rem, i32* @mod_resultsw, align 4
  %div = sdiv i32 %a, %b
  store i32 %div, i32* @div_resultsw, align 4
  ret void
; CHECK-LABEL: modulo_div_sw
; CHECK-NOT: modsw
; CHECK: div
; CHECK-NOT: modsw
; CHECK: mull
; CHECK-NOT: modsw
; CHECK: sub
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_div_sw
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind
define void @modulo_div_abc_sw(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr {
entry:
  %rem = srem i32 %a, %c
  store i32 %rem, i32* @mod_resultsw, align 4
  %div = sdiv i32 %b, %c
  store i32 %div, i32* @div_resultsw, align 4
  ret void
; CHECK-LABEL: modulo_div_abc_sw
; CHECK: modsw {{[0-9]+}}, 3, 5
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_div_abc_sw
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind
define void @modulo_div_uw(i32 zeroext %a, i32 zeroext %b) local_unnamed_addr {
entry:
  %rem = urem i32 %a, %b
  store i32 %rem, i32* @mod_resultuw, align 4
  %div = udiv i32 %a, %b
  store i32 %div, i32* @div_resultuw, align 4
  ret void
; CHECK-LABEL: modulo_div_uw
; CHECK-NOT: modsw
; CHECK: div
; CHECK-NOT: modsw
; CHECK: mull
; CHECK-NOT: modsw
; CHECK: sub
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_div_uw
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind
define void @modulo_div_swuw(i32 signext %a, i32 signext %b) local_unnamed_addr {
entry:
  %rem = srem i32 %a, %b
  store i32 %rem, i32* @mod_resultsw, align 4
  %div = udiv i32 %a, %b
  store i32 %div, i32* @div_resultsw, align 4
  ret void
; CHECK-LABEL: modulo_div_swuw
; CHECK: modsw {{[0-9]+}}, 3, 4
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_div_swuw
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind
define void @modulo_div_udsd(i64 %a, i64 %b) local_unnamed_addr {
entry:
  %rem = urem i64 %a, %b
  store i64 %rem, i64* @mod_resultud, align 8
  %div = sdiv i64 %a, %b
  store i64 %div, i64* @div_resultsd, align 8
  ret void
; CHECK-LABEL: modulo_div_udsd
; CHECK: modud {{[0-9]+}}, 3, 4
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_div_udsd
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind
define void @modulo_const32_sw(i32 signext %a) local_unnamed_addr {
entry:
  %rem = srem i32 %a, 32
  store i32 %rem, i32* @mod_resultsw, align 4
  ret void
; CHECK-LABEL: modulo_const32_sw
; CHECK-NOT: modsw
; CHECK: srawi
; CHECK-NOT: modsw
; CHECK: addze
; CHECK-NOT: modsw
; CHECK: slwi
; CHECK-NOT: modsw
; CHECK: subf
; CHECK-NOT: modsw
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_const32_sw
; CHECK-PWR8: srawi
; CHECK-PWR8: addze
; CHECK-PWR8: slwi
; CHECK-PWR8: subf
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @modulo_const3_sw(i32 signext %a) local_unnamed_addr {
entry:
  %rem = srem i32 %a, 3
  ret i32 %rem
; CHECK-LABEL: modulo_const3_sw
; CHECK-NOT: modsw
; CHECK: mulh
; CHECK-NOT: modsw
; CHECK: sub
; CHECK-NOT: modsw
; CHECK: blr
; CHECK-PWR8-LABEL: modulo_const3_sw
; CHECK-PWR8: mulh
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @const2_modulo_sw(i32 signext %a) local_unnamed_addr {
entry:
  %rem = srem i32 2, %a
  ret i32 %rem
; CHECK-LABEL: const2_modulo_sw
; CHECK: modsw {{[0-9]+}}, {{[0-9]+}}, 3
; CHECK: blr
; CHECK-PWR8-LABEL: const2_modulo_sw
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}

; Function Attrs: norecurse nounwind
; FIXME On power 9 this test will still produce modsw because the divide is in
; a different block than the remainder. Due to the nature of the SDAG we cannot
; see the div in the other block.
define void @blocks_modulo_div_sw(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr {
entry:
  %div = sdiv i32 %a, %b
  store i32 %div, i32* @div_resultsw, align 4
  %cmp = icmp sgt i32 %c, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %rem = srem i32 %a, %b
  store i32 %rem, i32* @mod_resultsw, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
; CHECK-LABEL: blocks_modulo_div_sw
; CHECK: div
; CHECK: modsw {{[0-9]+}}, 3, 4
; CHECK: blr
; CHECK-PWR8-LABEL: blocks_modulo_div_sw
; CHECK-PWR8: div
; CHECK-PWR8: mull
; CHECK-PWR8: sub
; CHECK-PWR8: blr
}


