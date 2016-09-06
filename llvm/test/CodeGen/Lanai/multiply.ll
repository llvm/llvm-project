; RUN: llc -march=lanai < %s | FileCheck %s

; Test the in place lowering of mul i32.

define i32 @f6(i32 inreg %a) #0 {
entry:
  %mul = mul nsw i32 %a, 6
  ret i32 %mul
}
; CHECK: sh %r6, 0x1, %r{{[0-9]+}}
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %rv

define i32 @f7(i32 inreg %a) #0 {
entry:
  %mul = mul nsw i32 %a, 7
  ret i32 %mul
}
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r6, %rv

define i32 @f8(i32 inreg %a) #0 {
entry:
  %mul = shl nsw i32 %a, 3
  ret i32 %mul
}
; CHECK: sh %r6, 0x3, %rv

define i32 @fm6(i32 inreg %a) #0 {
entry:
  %mul = mul nsw i32 %a, -6
  ret i32 %mul
}
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sh %r6, 0x1, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %rv

define i32 @fm7(i32 inreg %a) #0 {
entry:
  %mul = mul nsw i32 %a, -7
  ret i32 %mul
}
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r6, %r{{[0-9]+}}, %rv

define i32 @fm8(i32 inreg %a) #0 {
entry:
  %mul = mul nsw i32 %a, -8
  ret i32 %mul
}
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %rv

define i32 @h1(i32 inreg %a) #0 {
entry:
  %mul = mul i32 %a, -1431655765
  ret i32 %mul
}
; CHECK: h1
; CHECK: mulsi3
