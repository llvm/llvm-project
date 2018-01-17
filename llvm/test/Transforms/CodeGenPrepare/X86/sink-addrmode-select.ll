; RUN: opt -S -codegenprepare -disable-complex-addr-modes=false -addr-sink-new-select=true  %s | FileCheck %s --check-prefix=CHECK
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

; Select when both offset and scale reg are present.
define i64 @test1(i1 %c, i64* %b, i64 %scale) {
; CHECK-LABEL: @test1
entry:
; CHECK-LABEL: entry:
  %g = getelementptr inbounds i64, i64* %b, i64 %scale
  %g1 = getelementptr inbounds i64, i64* %g, i64 8
  %g2 = getelementptr inbounds i64, i64* %g, i64 16
  %s = select i1 %c, i64* %g1, i64* %g2
; CHECK-NOT: sunkaddr
  %v = load i64 , i64* %s, align 8
  ret i64 %v
}

