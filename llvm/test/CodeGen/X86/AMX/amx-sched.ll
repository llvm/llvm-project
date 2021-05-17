; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-int8 -mattr=+avx512f -mcpu=skx -verify-machineinstrs | FileCheck %s

define <256 x i32> @test_shape_sched(i16 %m, i16 %n, i16 %k, <256 x i32> %c, <256 x i32> %a, <256 x i32> %b) nounwind {
; Just to make sure shape def is not scheduled across ldtilecfg.
; CHECK-LABEL: test_shape_sched:
; CHECK:    ldtilecfg
; CHECK-NOT: movw
  %c1 = bitcast <256 x i32> %c to x86_amx
  %a1 = bitcast <256 x i32> %a to x86_amx
  %b1 = bitcast <256 x i32> %b to x86_amx
  %t = call x86_amx @llvm.x86.tdpbssd.internal(i16 %m, i16 %n, i16 %k, x86_amx %c1, x86_amx %a1, x86_amx %b1)
  %res = bitcast x86_amx %t to <256 x i32>
  ret <256 x i32> %res
}

define <256 x i32> @test_shape_sched2(i16 %m, i16 %n, i16 %k, i8* %c, i8* %a, i8* %b) nounwind {
; Just to make sure shape def is not scheduled across ldtilecfg.
; CHECK-LABEL: test_shape_sched2:
; CHECK:    ldtilecfg
; CHECK-NOT: movw
  %aa = lshr i16 %k, 2
  %c1 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %m, i16 %n, i8* %c, i64 64)
  %a1 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %m, i16 %k, i8* %a, i64 64)
  %b1 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %aa, i16 %n, i8* %b, i64 64)
  %t = call x86_amx @llvm.x86.tdpbssd.internal(i16 %m, i16 %n, i16 %k, x86_amx %c1, x86_amx %a1, x86_amx %b1)
  %res = bitcast x86_amx %t to <256 x i32>
  ret <256 x i32> %res
}

declare x86_amx @llvm.x86.tileloadd64.internal(i16, i16, i8*, i64)
declare x86_amx @llvm.x86.tdpbssd.internal(i16, i16, i16, x86_amx, x86_amx, x86_amx)
