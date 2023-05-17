; RUN: llc < %s -mtriple=thumbv7-apple-ios | FileCheck %s

define void @bar(ptr %p, i32 %lane, <4 x i32> %phitmp) nounwind {
; CHECK:  lsls r[[ADDR:[0-9]+]], r[[ADDR]], #2
; CHECK:  vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[SOURCE:[0-9]+]]:128], r[[ADDR]]
; CHECK:  vld1.32 {[[DREG:d[0-9]+]][], [[DREG2:d[0-9]+]][]}, [r[[SOURCE]]:32]
; CHECK:  vst1.32 {[[DREG]], [[DREG2]]}, [r0]
  %val = extractelement <4 x i32> %phitmp, i32 %lane
  %r1 = insertelement <4 x i32> undef, i32 %val, i32 1
  %r2 = insertelement <4 x i32> %r1, i32 %val, i32 2
  %r3 = insertelement <4 x i32> %r2, i32 %val, i32 3
  store <4 x i32> %r3, ptr %p, align 4
  ret void
}
