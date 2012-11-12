target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=2 -instcombine -gvn -S | FileCheck %s

; Make sure that things (specifically getelementptr) are not connected to loads
; and stores via the address operand (which would be bad because the address
; is really a scalar even after vectorization)
define i64 @test2(i64 %a) nounwind uwtable readonly {
entry:
  %a1 = inttoptr i64 %a to i64*
  %a2 = getelementptr i64* %a1, i64 1
  %a3 = getelementptr i64* %a1, i64 2
  %v2 = load i64* %a2, align 8
  %v3 = load i64* %a3, align 8
  %v2a = add i64 %v2, 5
  %v3a = add i64 %v3, 7
  store i64 %v2a, i64* %a2, align 8
  store i64 %v3a, i64* %a3, align 8
  %r = add i64 %v2, %v3
  ret i64 %r
; CHECK: @test2
; CHECK-NOT: getelementptr <2 x i64*>
}

