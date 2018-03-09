; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that we generate zero-extends, instead of just shifting and oring
; registers (which can contain sign-extended negative values).
; CHECK: and(r{{[0-9]+}},#255)

define i32 @fred(i8 %a0, i8 %a1, i8 %a2, i8 %a3) #0 {
b4:
  %v5 = insertelement <4 x i8> undef, i8 %a0, i32 0
  %v6 = insertelement <4 x i8> %v5, i8 %a1, i32 1
  %v7 = insertelement <4 x i8> %v6, i8 %a2, i32 2
  %v8 = insertelement <4 x i8> %v7, i8 %a3, i32 3
  %v9 = bitcast <4 x i8> %v8 to i32
  ret i32 %v9
}

attributes #0 = { nounwind readnone }
