; RUN: llc -verify-machineinstrs -mcpu=pwr7 -code-model=medium <%s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -code-model=large <%s | FileCheck -check-prefix=LARGE  %s

; Test correct code generation for medium and large code model
; for loading the address of a jump table from the TOC.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define signext i32 @test_jump_table(i32 signext %i) nounwind {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  switch i32 %0, label %sw.default [
    i32 3, label %sw.bb
    i32 4, label %sw.bb1
    i32 5, label %sw.bb2
    i32 6, label %sw.bb3
  ]

sw.default:                                       ; preds = %entry
  br label %sw.epilog

sw.bb:                                            ; preds = %entry
  %1 = load i32, i32* %i.addr, align 4
  %mul = mul nsw i32 %1, 7
  store i32 %mul, i32* %i.addr, align 4
  br label %sw.bb1

sw.bb1:                                           ; preds = %entry, %sw.bb
  %2 = load i32, i32* %i.addr, align 4
  %dec = add nsw i32 %2, -1
  store i32 %dec, i32* %i.addr, align 4
  br label %sw.bb2

sw.bb2:                                           ; preds = %entry, %sw.bb1
  %3 = load i32, i32* %i.addr, align 4
  %add = add nsw i32 %3, 3
  store i32 %add, i32* %i.addr, align 4
  br label %sw.bb3

sw.bb3:                                           ; preds = %entry, %sw.bb2
  %4 = load i32, i32* %i.addr, align 4
  %shl = shl i32 %4, 1
  store i32 %shl, i32* %i.addr, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb3, %sw.default
  %5 = load i32, i32* %i.addr, align 4
  ret i32 %5
}
; CHECK-LABEL: test_jump_table:
; CHECK-NOT:       bl .L0$pb

; CHECK:       addis [[REG1:[0-9]+]], 2, .LC[[TOCNUM:[0-9]+]]@toc@ha
; CHECK:       ld [[REG2:[0-9]+]], .LC[[TOCNUM]]@toc@l([[REG1]])
; CHECK:       lwax [[REG3:[0-9]+]], {{[0-9]+}}, [[REG2]]
; CHECK-NEXT:  add  [[REG4:[0-9]+]], [[REG3]], [[REG2]]
; CHECK-NEXT:  mtctr [[REG4]]
; CHECK-NEXT:  bctr

; CHECK-LABEL: .LJTI0_0:
; CHECK-NEXT: .long	.LBB0_{{[0-9]+}}-.LJTI0_0

; LARGE-LABEL: test_jump_table:
; LARGE:       bl .L0$pb
; LARGE-NEXT:  .L0$pb:
; LARGE:       mflr [[REGBASE:[0-9]+]]

; LARGE:       addis [[REG1:[0-9]+]], 2, .LC[[TOCNUM:[0-9]+]]@toc@ha
; LARGE:       ld [[REG2:[0-9]+]], .LC[[TOCNUM]]@toc@l([[REG1]])
; LARGE:       lwax [[REG3:[0-9]+]], {{[0-9]+}}, [[REG2]]
; LARGE-NEXT:  add  [[REG4:[0-9]+]], [[REG3]], [[REGBASE]]
; LARGE-NEXT:  mtctr [[REG4]]
; LARGE-NEXT:  bctr

; LARGE-LABEL: .LJTI0_0:
; LARGE-NEXT: .long	.LBB0_{{[0-9]+}}-.L0$pb
