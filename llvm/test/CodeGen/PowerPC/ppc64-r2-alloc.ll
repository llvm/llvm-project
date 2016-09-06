; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define signext i32 @foo(i32 signext %a, i32 signext %d) #0 {
entry:
  %div = sdiv i32 %a, %d
  %div1 = sdiv i32 %div, %d
  %div2 = sdiv i32 %div1, %d
  %div3 = sdiv i32 %div2, %d
  %div4 = sdiv i32 %div3, %d
  %div5 = sdiv i32 %div4, %d
  %div6 = sdiv i32 %div5, %d
  %div7 = sdiv i32 %div6, %d
  %div8 = sdiv i32 %div7, %d
  %div9 = sdiv i32 %div8, %d
  %div10 = sdiv i32 %div9, %d
  %div11 = sdiv i32 %div10, %d
  %div12 = sdiv i32 %div11, %d
  %div13 = sdiv i32 %div12, %d
  %div14 = sdiv i32 %div13, %d
  %div15 = sdiv i32 %div14, %d
  %div16 = sdiv i32 %div15, %d
  %div17 = sdiv i32 %div16, %d
  %div18 = sdiv i32 %div17, %d
  %div19 = sdiv i32 %div18, %d
  %div20 = sdiv i32 %div19, %d
  %div21 = sdiv i32 %div20, %d
  %div22 = sdiv i32 %div21, %d
  %div23 = sdiv i32 %div22, %d
  %div24 = sdiv i32 %div23, %d
  %div25 = sdiv i32 %div24, %d
  %div26 = sdiv i32 %div25, %d
  %div27 = sdiv i32 %div26, %d
  %div28 = sdiv i32 %div27, %d
  %div29 = sdiv i32 %div28, %d
  %div30 = sdiv i32 %div29, %d
  %div31 = sdiv i32 %div30, %d
  %div32 = sdiv i32 %div31, %d
  %div33 = sdiv i32 %div32, %div31
  %div34 = sdiv i32 %div33, %div30
  %div35 = sdiv i32 %div34, %div29
  %div36 = sdiv i32 %div35, %div28
  %div37 = sdiv i32 %div36, %div27
  %div38 = sdiv i32 %div37, %div26
  %div39 = sdiv i32 %div38, %div25
  %div40 = sdiv i32 %div39, %div24
  %div41 = sdiv i32 %div40, %div23
  %div42 = sdiv i32 %div41, %div22
  %div43 = sdiv i32 %div42, %div21
  %div44 = sdiv i32 %div43, %div20
  %div45 = sdiv i32 %div44, %div19
  %div46 = sdiv i32 %div45, %div18
  %div47 = sdiv i32 %div46, %div17
  %div48 = sdiv i32 %div47, %div16
  %div49 = sdiv i32 %div48, %div15
  %div50 = sdiv i32 %div49, %div14
  %div51 = sdiv i32 %div50, %div13
  %div52 = sdiv i32 %div51, %div12
  %div53 = sdiv i32 %div52, %div11
  %div54 = sdiv i32 %div53, %div10
  %div55 = sdiv i32 %div54, %div9
  %div56 = sdiv i32 %div55, %div8
  %div57 = sdiv i32 %div56, %div7
  %div58 = sdiv i32 %div57, %div6
  %div59 = sdiv i32 %div58, %div5
  %div60 = sdiv i32 %div59, %div4
  %div61 = sdiv i32 %div60, %div3
  %div62 = sdiv i32 %div61, %div2
  %div63 = sdiv i32 %div62, %div1
  %div64 = sdiv i32 %div63, %div
  ret i32 %div64
}

; This function will need to use all non-reserved GPRs (and then some), make
; sure that r2 is among them.
; CHECK-LABEL: @foo
; CHECK: std 2,
; CHECK: ld 2,
; CHECK: blr

