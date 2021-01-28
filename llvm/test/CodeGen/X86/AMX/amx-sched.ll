; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-int8 -mattr=+avx512f -mcpu=skx -verify-machineinstrs | FileCheck %s

define <256 x i32> @test_shape_sched(i16 %m, i16 %n, i16 %k, <256 x i32> %c, <256 x i32> %a, <256 x i32> %b) nounwind {
; Just to make sure shape def is not scheduled across ldtilecfg.
; CHECK:    ldtilecfg
; CHECK-NOT: movw
  %c1 = bitcast <256 x i32> %c to x86_amx
  %a1 = bitcast <256 x i32> %a to x86_amx
  %b1 = bitcast <256 x i32> %b to x86_amx
  %t = call x86_amx @llvm.x86.tdpbssd.internal(i16 %m, i16 %n, i16 %k, x86_amx %c1, x86_amx %a1, x86_amx %b1)
  %res = bitcast x86_amx %t to <256 x i32>
  ret <256 x i32> %res
}


declare x86_amx @llvm.x86.tdpbssd.internal(i16, i16, i16, x86_amx, x86_amx, x86_amx)
