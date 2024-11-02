; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s 
; REQUIRES: asserts

define protected swiftcc void @"$s22LanguageServerProtocol13HoverResponseV8contents5rangeAcA13MarkupContentV_SnyAA8PositionVGSgtcfC"() {
  %1 = load <2 x i64>, ptr undef, align 16
  %2 = load i1, ptr undef, align 8
  %3 = insertelement <2 x i1> undef, i1 %2, i32 0
  %4 = shufflevector <2 x i1> %3, <2 x i1> undef, <2 x i32> zeroinitializer
  %5 = select <2 x i1> %4, <2 x i64> zeroinitializer, <2 x i64> %1
  store <2 x i64> %5, ptr undef, align 8
  ret void
}
