; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:64:64-i128:64-v128:64:128-a:0:64-n64-S64"
target triple = "thumbv6---gnueabi"

; Function Attrs: norecurse nounwind readonly
define i128 @a(i64* nocapture readonly %z) local_unnamed_addr #0 {
entry:
  %0 = load i64, i64* %z, align 4
  %conv.i = zext i64 %0 to i128
  %arrayidx1 = getelementptr inbounds i64, i64* %z, i64 2
  %1 = load i64, i64* %arrayidx1, align 4
  %conv.i38 = zext i64 %1 to i128
  %shl.i39 = shl nuw i128 %conv.i38, 64
  %or = or i128 %shl.i39, %conv.i
  %arrayidx3 = getelementptr inbounds i64, i64* %z, i64 1
  %2 = load i64, i64* %arrayidx3, align 4
  %conv.i37 = zext i64 %2 to i128
  %arrayidx5 = getelementptr inbounds i64, i64* %z, i64 3
  %3 = load i64, i64* %arrayidx5, align 4
  %conv.i35 = zext i64 %3 to i128
  %shl.i36 = shl nuw i128 %conv.i35, 64
  %or7 = or i128 %shl.i36, %conv.i37
  %arrayidx10 = getelementptr inbounds i64, i64* %z, i64 4
  %4 = load i64, i64* %arrayidx10, align 4
  %conv.i64 = zext i64 %4 to i128
  %shl.i33 = shl nuw i128 %conv.i64, 64
  %or12 = or i128 %shl.i33, %conv.i
  %arrayidx15 = getelementptr inbounds i64, i64* %z, i64 5
  %5 = load i64, i64* %arrayidx15, align 4
  %conv.i30 = zext i64 %5 to i128
  %shl.i = shl nuw i128 %conv.i30, 64
  %or17 = or i128 %shl.i, %conv.i37
  %add = add i128 %or7, %or
  %add18 = add i128 %or17, %or12
  %mul = mul i128 %add18, %add
  ret i128 %mul
}
; CHECK: adds r5, r1, r6
; CHECK: mov r5, r4
; CHECK: adcs r5, r7
; CHECK: ldr r5, [sp, #12]                   @ 4-byte Reload
; CHECK: adcs r2, r5
; CHECK: ldr r5, [sp, #16]                   @ 4-byte Reload
; CHECK: adcs r3, r5
; CHECK: adds r6, r1, r6
; CHECK: adcs r4, r7
; CHECK: ldr r1, [r0, #20]
; CHECK: str r1, [sp, #16]                   @ 4-byte Spill
; CHECK: ldr r5, [r0, #28]
; CHECK: ldr r1, [r0, #16]
; CHECK: ldr r7, [r0, #24]
; CHECK: adcs r7, r1
; CHECK: ldr r0, [sp, #16]                   @ 4-byte Reload
; CHECK: adcs r5, r0
