; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv5e-none-linux-gnueabi"

; Function Attrs: norecurse nounwind optsize
define void @f(i32,i32,i32,i32,ptr %x4p, ptr %x5p, ptr %x6p) {
if.end:
  br label %while.body

while.body:
  %ll.0100 = phi i64 [ 0, %if.end ], [ %shr32, %while.body ]
  %add = add nuw nsw i64 %ll.0100, 0
  %add3 = add nuw nsw i64 %add, 0
  %shr = lshr i64 %add3, 32
  %conv7 = zext i32 %0 to i64
  %conv9 = zext i32 %1 to i64
  %add10 = add nuw nsw i64 %conv9, %conv7
  %add11 = add nuw nsw i64 %add10, %shr
  %shr14 = lshr i64 %add11, 32
  %conv16 = zext i32 %2 to i64
  %conv18 = zext i32 %3 to i64
  %add19 = add nuw nsw i64 %conv18, %conv16
  %add20 = add nuw nsw i64 %add19, %shr14
  %conv21 = trunc i64 %add20 to i32
  store i32 %conv21, ptr %x6p, align 4
  %shr23 = lshr i64 %add20, 32
  %x4 = load i32, ptr %x4p, align 4
  %conv25 = zext i32 %x4 to i64
  %x5 = load i32, ptr %x5p, align 4
  %conv27 = zext i32 %x5 to i64
  %add28 = add nuw nsw i64 %conv27, %conv25
  %add29 = add nuw nsw i64 %add28, %shr23
  %shr32 = lshr i64 %add29, 32
  br label %while.body
}
; CHECK: adds	r5, r0, r1
; CHECK: push	{r3}
; CHECK: pop	{r1}
; CHECK: adcs	r1, r3
; CHECK: ldr	r0, [sp, #12]           @ 4-byte Reload
; CHECK: ldr	r2, [sp, #8]            @ 4-byte Reload
; CHECK: adds	r2, r0, r2
; CHECK: push	{r3}
; CHECK: pop	{r4}
; CHECK: adcs	r4, r3
; CHECK: adds	r0, r2, r3
; CHECK: push	{r5}
; CHECK: pop	{r0}
; CHECK: adcs	r0, r4
; CHECK: ldr	r6, [sp, #4]            @ 4-byte Reload
; CHECK: str	r0, [r6]
; CHECK: ldr	r0, [r7]
; CHECK: ldr	r6, [sp]                @ 4-byte Reload
; CHECK: ldr	r6, [r6]
; CHECK: adds	r0, r6, r0
