; RUN: llc < %s -march=x86-64 -mcpu=corei7 -mtriple=x86_64-pc-win32 | FileCheck %s

;CHECK: vcast
define <2 x i32> @vcast(<2 x float> %a, <2 x float> %b) {
;CHECK: pmovzxdq
;CHECK: pmovzxdq
  %af = bitcast <2 x float> %a to <2 x i32>
  %bf = bitcast <2 x float> %b to <2 x i32>
  %x = sub <2 x i32> %af, %bf
;CHECK: psubq
  ret <2 x i32> %x
;CHECK: ret
}

