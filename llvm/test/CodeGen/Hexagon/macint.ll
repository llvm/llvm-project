; RUN: llc -march=hexagon -mcpu=hexagonv4  < %s | FileCheck %s
; Check that we generate integer multiply accumulate.

; CHECK: r{{[0-9]+}} += mpyi(r{{[0-9]+}}, r{{[0-9]+}})

define i32 @main(i32* %a, i32* %b) nounwind {
  entry:
  %0 = load i32* %a, align 4
  %div = udiv i32 %0, 10000
  %rem = urem i32 %div, 10
  store i32 %rem, i32* %b, align 4
  ret i32 0
}

