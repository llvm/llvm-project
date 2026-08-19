; RUN: llc -o - %s | FileCheck %s
define i8 @test_ucmp_i8_i1(i1 zeroext %a, i1 zeroext %b) {
  %cmp = call i8 @llvm.ucmp.i8.i1(i1 %a, i1 %b)
  ret i8 %cmp
}

define i16 @test_ucmp_i16_i1(i1 zeroext %a, i1 zeroext %b) {
  %cmp = call i16 @llvm.ucmp.i16.i1(i1 %a, i1 %b)
  ret i16 %cmp
}

declare i8 @llvm.ucmp.i8.i1(i1, i1)
declare i16 @llvm.ucmp.i16.i1(i1, i1)
