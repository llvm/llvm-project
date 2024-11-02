; RUN: llc -march=hexagon < %s | FileCheck %s

; This used to crash with "cannot select" error.
; CHECK: vlsrh(r1:0,#4)

target triple = "hexagon-unknown-linux-gnu"

define <2 x i16> @foo(ptr nocapture %v) nounwind {
  %vec = load <2 x i32>, ptr %v, align 8
  %trunc = trunc <2 x i32> %vec to <2 x i16>
  %r = lshr <2 x i16> %trunc, <i16 4, i16 4>
  ret <2 x i16> %r
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }

