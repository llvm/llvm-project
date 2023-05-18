; RUN: llc -O2 -mcpu=hexagonv5 < %s | FileCheck %s
; There should be no register pair used.
; CHECK-NOT: r{{.*}}:{{[0-9]}} = and
; CHECK-NOT: r{{.*}}:{{[0-9]}} = xor
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define i32 @foo(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
  %and = and i64 %y, -361700868401135616
  %xor = xor i64 %and, %z
  %shr1 = lshr i64 %xor, 32
  %conv = trunc i64 %shr1 to i32
  ret i32 %conv
}
