; RUN: llc < %s -mtriple=thumbv6m-eabi -verify-machineinstrs -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-none--eabi"

@a = external global i32*
@b = external global i32*

; Function Attrs: nounwind
define void @foo24() #0 {
entry:
; CHECK-LABEL: foo24:
; CHECK: ldr r[[LB:[0-9]]], .LCPI
; CHECK: adds r[[NLB:[0-9]]], r[[LB]], #4
; CHECK: ldr r[[SB:[0-9]]], .LCPI
; CHECK: adds r[[NSB:[0-9]]], r[[SB]], #4
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]]}
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]]}
  %0 = load i32*, i32** @a, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 1
  %1 = bitcast i32* %arrayidx to i8*
  %2 = load i32*, i32** @b, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = bitcast i32* %arrayidx1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %3, i32 24, i1 false)
  ret void
}

define void @foo28() #0 {
entry:
; CHECK-LABEL: foo28:
; CHECK: ldr r[[LB:[0-9]]], .LCPI
; CHECK: adds r[[NLB:[0-9]]], r[[LB]], #4
; CHECK: ldr r[[SB:[0-9]]], .LCPI
; CHECK: adds r[[NSB:[0-9]]], r[[SB]], #4
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]]}
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]], r[[R4:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]], r[[R4]]}
  %0 = load i32*, i32** @a, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 1
  %1 = bitcast i32* %arrayidx to i8*
  %2 = load i32*, i32** @b, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = bitcast i32* %arrayidx1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %3, i32 28, i1 false)
  ret void
}

define void @foo32() #0 {
entry:
; CHECK-LABEL: foo32:
; CHECK: ldr r[[LB:[0-9]]], .LCPI
; CHECK: adds r[[NLB:[0-9]]], r[[LB]], #4
; CHECK: ldr r[[SB:[0-9]]], .LCPI
; CHECK: adds r[[NSB:[0-9]]], r[[SB]], #4
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]], r[[R4:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]], r[[R4]]}
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]], r[[R4:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]], r[[R4]]}
  %0 = load i32*, i32** @a, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 1
  %1 = bitcast i32* %arrayidx to i8*
  %2 = load i32*, i32** @b, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = bitcast i32* %arrayidx1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %3, i32 32, i1 false)
  ret void
}

define void @foo36() #0 {
entry:
; CHECK-LABEL: foo36:
; CHECK: ldr r[[LB:[0-9]]], .LCPI
; CHECK: adds r[[NLB:[0-9]]], r[[LB]], #4
; CHECK: ldr r[[SB:[0-9]]], .LCPI
; CHECK: adds r[[NSB:[0-9]]], r[[SB]], #4
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]]}
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]]}
; CHECK-NEXT: ldm r[[NLB]]!, {r[[R1:[0-9]]], r[[R2:[0-9]]], r[[R3:[0-9]]]}
; CHECK-NEXT: stm r[[NSB]]!, {r[[R1]], r[[R2]], r[[R3]]}
  %0 = load i32*, i32** @a, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 1
  %1 = bitcast i32* %arrayidx to i8*
  %2 = load i32*, i32** @b, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = bitcast i32* %arrayidx1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %3, i32 36, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1) #1
