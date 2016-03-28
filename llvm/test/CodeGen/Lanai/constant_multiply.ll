; RUN: llc < %s | FileCheck %s

; Test custom lowering for 32-bit integer multiplication.

target datalayout = "E-m:e-p:32:32-i64:64-a:0:32-n32-S64"
target triple = "lanai"

; CHECK-LABEL: f6:
; CHECK: sh %r6, 0x1, %r{{[0-9]+}}
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @f6(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, 6
  ret i32 %1
}

; CHECK-LABEL: f7:
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r6, %rv
define i32 @f7(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, 7
  ret i32 %1
}

; CHECK-LABEL: f8:
; CHECK: sh %r6, 0x3, %rv
define i32 @f8(i32 inreg %a) #0 {
  %1 = shl nsw i32 %a, 3
  ret i32 %1
}

; CHECK-LABEL: f9:
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: add %r{{[0-9]+}}, %r6, %rv
define i32 @f9(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, 9
  ret i32 %1
}

; CHECK-LABEL: f10:
; CHECK: sh %r6, 0x1, %r{{[0-9]+}}
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: add %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @f10(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, 10
  ret i32 %1
}

; CHECK-LABEL: f1280:
; CHECK: sh %r6, 0x8, %r{{[0-9]+}}
; CHECK: sh %r6, 0xa, %r{{[0-9]+}}
; CHECK: add %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @f1280(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, 1280
  ret i32 %1
}

; CHECK-LABEL: fm6:
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sh %r6, 0x1, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @fm6(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, -6
  ret i32 %1
}

; CHECK-LABEL: fm7:
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r6, %r{{[0-9]+}}, %rv
define i32 @fm7(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, -7
  ret i32 %1
}

; CHECK-LABEL: fm8:
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @fm8(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, -8
  ret i32 %1
}

; CHECK-LABEL: fm9:
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r6, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @fm9(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, -9
  ret i32 %1
}

; CHECK-LABEL: fm10:
; CHECK: sh %r6, 0x3, %r{{[0-9]+}}
; CHECK: sh %r6, 0x1, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: sub %r{{[0-9]+}}, %r{{[0-9]+}}, %rv
define i32 @fm10(i32 inreg %a) #0 {
  %1 = mul nsw i32 %a, -10
  ret i32 %1
}

; CHECK-LABEL: h1:
; CHECK: __mulsi3
define i32 @h1(i32 inreg %a) #0 {
  %1 = mul i32 %a, -1431655765
  ret i32 %1
}
