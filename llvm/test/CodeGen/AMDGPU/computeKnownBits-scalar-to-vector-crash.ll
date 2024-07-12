; RUN: llc -mtriple=amdgcn -mcpu=gfx802 -verify-machineinstrs < %s | FileCheck %s

; CHECK: s_waitcnt
define <2 x i16> @main(<2 x float>) #0 {
  %2 = bitcast <2 x float> %0 to <4 x i16>
  %3 = shufflevector <4 x i16> %2, <4 x i16> undef, <2 x i32> <i32 0, i32 undef>
  %4 = extractelement <4 x i16> %2, i32 0
  %5 = insertelement <2 x i16> %3, i16 %4, i32 0
  ret <2 x i16> %5
}

