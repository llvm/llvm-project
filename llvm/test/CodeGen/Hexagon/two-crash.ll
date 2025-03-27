; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; This testcase crashed, because we propagated a reg:sub into a tied use.
; The two-address pass rewrote it in a way that generated incorrect code.
; CHECK: r{{[0-9]+}} += lsr(r{{[0-9]+}},#16)

target triple = "hexagon"

define i64 @fred(i64 %x) local_unnamed_addr #0 {
entry:
  %t.sroa.0.0.extract.trunc = trunc i64 %x to i32
  %t4.sroa.4.0.extract.shift = lshr i64 %x, 16
  %add11 = add i32 0, %t.sroa.0.0.extract.trunc
  %t14.sroa.3.0.extract.trunc = trunc i64 %t4.sroa.4.0.extract.shift to i32
  %t14.sroa.4.0.extract.shift = lshr i64 %x, 24
  %add21 = add i32 %add11, %t14.sroa.3.0.extract.trunc
  %t24.sroa.3.0.extract.trunc = trunc i64 %t14.sroa.4.0.extract.shift to i32
  %add31 = add i32 %add21, %t24.sroa.3.0.extract.trunc
  %conv32.mask = and i32 %add31, 255
  %conv33 = zext i32 %conv32.mask to i64
  ret i64 %conv33
}

attributes #0 = { norecurse nounwind readnone }
