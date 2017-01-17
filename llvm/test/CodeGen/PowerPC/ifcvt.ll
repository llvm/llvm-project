; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -verify-machineinstrs -ppc-gen-isel=false | FileCheck --check-prefix=CHECK-NO-ISEL %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @test(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %sext82 = shl i32 %d, 16
  %conv29 = ashr exact i32 %sext82, 16
  %cmp = icmp slt i32 %sext82, 0
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %sw.epilog
  %and33 = and i32 %conv29, 32767
  %sub34 = sub nsw i32 %a, %and33
  br label %cond.end

cond.false:                                       ; preds = %sw.epilog
  %add37 = add nsw i32 %conv29, %a
  br label %cond.end

; CHECK-LABEL: @test
; CHECK-NO-ISEL-LABEL: @test
; CHECK: add [[REG:[0-9]+]], 
; CHECK: subf [[REG2:[0-9]+]],
; CHECK: isel {{[0-9]+}}, [[REG]], [[REG2]],
; CHECK-NO-ISEL: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NO-ISEL: [[TRUE]]
; CHECK-NO-ISEL-NEXT: addi 5, 6, 0
; CHECK-NO-ISEL: extsh 5, 5
; CHECK-NO-ISEL-NEXT: add 3, 3, 5
; CHECK-NO-ISEL-NEXT: blr

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %sub34, %cond.true ], [ %add37, %cond.false ]
  %sext83 = shl i32 %cond, 16
  %conv39 = ashr exact i32 %sext83, 16
  %add41 = sub i32 %b, %a
  %sub43 = add i32 %add41, %conv39
  ret i32 %sub43
}

