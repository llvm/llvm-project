; RUN: llc -mtriple=x86_64-apple-macosx10 -mattr=avx -show-mc-encoding < %s | FileCheck %s

define i64 @t1(double %d_ivar) nounwind uwtable ssp {
entry:
; CHECK: t1
  %0 = bitcast double %d_ivar to i64
; CHECK: vmovd
; CHECK: encoding: [0xc4,0xe1,0xf9,0x7e,0xc0]
  ret i64 %0
}

define double @t2(i64 %d_ivar) nounwind uwtable ssp {
entry:
; CHECK: t2
  %0 = bitcast i64 %d_ivar to double
; CHECK: vmovd
; CHECK: encoding: [0xc4,0xe1,0xf9,0x6e,0xc7]
  ret double %0
}
