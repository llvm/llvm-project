; RUN: opt -passes=indvars -S < %s | FileCheck %s

declare { i8, i1 } @llvm.smul.with.overflow.i8(i8, i8) nounwind readnone

define i1 @test1(i8 %x) {
 entry:
; CHECK-LABEL: @test1
  %rem = srem i8 %x, 15
  %t = call { i8, i1 } @llvm.smul.with.overflow.i8(i8 %rem, i8 %rem)
; CHECK: %t = call { i8, i1 } @llvm.smul.with.overflow.i8(i8 %rem, i8 %rem)
; CHECK: %obit = extractvalue { i8, i1 } %t, 1
; CHECK: ret i1 %obit
  %obit = extractvalue { i8, i1 } %t, 1
  ret i1 %obit
}
