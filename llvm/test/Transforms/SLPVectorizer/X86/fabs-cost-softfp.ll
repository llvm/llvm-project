; Regression test from https://bugs.llvm.org/show_bug.cgi?id=39168
; Based on code from `compiler-rt/lib/builtins/multc3.c`
; On plaforms where fp128 lowers to an interger type (soft-fp) we
; shouldn't be calling isFAbsFree() on the legalized type.

; RUN: opt -slp-vectorizer -slp-threshold=-10 -S %s | FileCheck %s
; CHECK: call <2 x fp128> @llvm.fabs.v2f128(<2 x fp128

target triple = "i686-unknown-linux-gnu"

define void @vectorize_fp128(fp128 %c, fp128 %d) #0 {
entry:
  %0 = tail call fp128 @llvm.fabs.f128(fp128 %c)
  %cmpinf10 = fcmp oeq fp128 %0, 0xL00000000000000007FFF000000000000
  %1 = tail call fp128 @llvm.fabs.f128(fp128 %d)
  %cmpinf12 = fcmp oeq fp128 %1, 0xL00000000000000007FFF000000000000
  %or.cond39 = or i1 %cmpinf10, %cmpinf12
  br i1 %or.cond39, label %if.then13, label %if.end24

if.then13:                                        ; preds = %entry
  unreachable

if.end24:                                         ; preds = %entry
  ret void
}

declare fp128 @llvm.fabs.f128(fp128)

attributes #0 = { "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" }
