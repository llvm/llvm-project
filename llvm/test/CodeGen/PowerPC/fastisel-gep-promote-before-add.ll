; fastisel should not fold add with non-pointer bitwidth
; sext(a) + sext(b) != sext(a + b)
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-freebsd10.0 %s -O0 -o - | FileCheck %s

define zeroext i8 @gep_promotion(i8* %ptr) nounwind {
entry:
  %ptr.addr = alloca i8*, align 8
  %add = add i8 64, 64 ; 0x40 + 0x40
  %0 = load i8*, i8** %ptr.addr, align 8

  ; CHECK-LABEL: gep_promotion:
  ; CHECK: lbz {{[0-9]+}}, 0({{.*}})
  %arrayidx = getelementptr inbounds i8, i8* %0, i8 %add

  %1 = load i8, i8* %arrayidx, align 1
  ret i8 %1
}
