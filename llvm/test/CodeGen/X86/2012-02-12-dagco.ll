; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx
target triple = "x86_64-unknown-linux-gnu"
; Make sure we are not crashing on this one
define void @dagco_crash() {
entry:
  %srcval.i411.i = load <4 x i64>* undef, align 1
  %0 = extractelement <4 x i64> %srcval.i411.i, i32 3
  %srcval.i409.i = load <2 x i64>* undef, align 1
  %1 = extractelement <2 x i64> %srcval.i409.i, i32 0
  %2 = insertelement <8 x i64> undef, i64 %0, i32 5
  %3 = insertelement <8 x i64> %2, i64 %1, i32 6
  %4 = insertelement <8 x i64> %3, i64 undef, i32 7
  store <8 x i64> %4, <8 x i64> addrspace(1)* undef, align 64
  unreachable
}

