; RUN: llc < %s -march=x86-64 -mcpu=nehalem -mattr=+sse42 | FileCheck %s
; widen extract subvector

define void @convert(<2 x double>* %dst.addr, <3 x double> %src)  {
entry:
; CHECK: convert:
; CHECK: unpcklpd {{%xmm[0-7]}}, {{%xmm[0-7]}}
; CHECK-NEXT: movapd
  %val = shufflevector <3 x double> %src, <3 x double> undef, <2 x i32> < i32 0, i32 1>
  store <2 x double> %val, <2 x double>* %dst.addr
  ret void
}
