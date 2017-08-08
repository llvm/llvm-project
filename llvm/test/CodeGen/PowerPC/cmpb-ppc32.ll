; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define zeroext i16 @test16(i16 zeroext %x, i16 zeroext %y) #0 {
entry:
  %0 = xor i16 %y, %x
  %1 = and i16 %0, 255
  %cmp = icmp eq i16 %1, 0
  %cmp20 = icmp ult i16 %0, 256
  %conv25 = select i1 %cmp, i32 255, i32 0
  %conv27 = select i1 %cmp20, i32 65280, i32 0
  %or = or i32 %conv25, %conv27
  %conv29 = trunc i32 %or to i16
  ret i16 %conv29

; CHECK-LABEL: @test16
; CHECK: cmpb [[REG1:[0-9]+]], 4, 3
; CHECK: clrlwi 3, [[REG1]], 16
; CHECK: blr
}

define i32 @test32(i32 %x, i32 %y) #0 {
entry:
  %0 = xor i32 %y, %x
  %1 = and i32 %0, 255
  %cmp = icmp eq i32 %1, 0
  %2 = and i32 %0, 65280
  %cmp28 = icmp eq i32 %2, 0
  %3 = and i32 %0, 16711680
  %cmp34 = icmp eq i32 %3, 0
  %cmp40 = icmp ult i32 %0, 16777216
  %conv44 = select i1 %cmp, i32 255, i32 0
  %conv45 = select i1 %cmp28, i32 65280, i32 0
  %conv47 = select i1 %cmp34, i32 16711680, i32 0
  %conv50 = select i1 %cmp40, i32 -16777216, i32 0
  %or = or i32 %conv45, %conv50
  %or49 = or i32 %or, %conv44
  %or52 = or i32 %or49, %conv47
  ret i32 %or52

; CHECK-LABEL: @test32
; CHECK: cmpb 3, 4, 3
; CHECK-NOT: rlwinm
; CHECK: blr
}

attributes #0 = { nounwind readnone }

