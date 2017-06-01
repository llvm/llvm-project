; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-gnu-linux  < %s | FileCheck %s -check-prefix=CHECK

; Check size 8
; Function Attrs: nounwind readonly
define signext i32 @test1(i32* nocapture readonly %buffer1, i32* nocapture readonly %buffer2) local_unnamed_addr #0 {
entry:
  %0 = bitcast i32* %buffer1 to i8*
  %1 = bitcast i32* %buffer2 to i8*
  %call = tail call signext i32 @memcmp(i8* %0, i8* %1, i64 8) #2
  ret i32 %call

; CHECK-LABEL: @test1
; CHECK: ldbrx [[LOAD1:[0-9]+]]
; CHECK-NEXT: ldbrx [[LOAD2:[0-9]+]]
; CHECK-NEXT: li [[LI:[0-9]+]], 1
; CHECK-NEXT: cmpld [[CMPLD:[0-9]+]], [[LOAD1]], [[LOAD2]]
; CHECK-NEXT: subf. [[SUB:[0-9]+]], [[LOAD2]], [[LOAD1]]
; CHECK-NEXT: li [[LI2:[0-9]+]], -1
; CHECK-NEXT: isel [[ISEL:[0-9]+]], [[LI2]], [[LI]], 4
; CHECK-NEXT: isel [[ISEL2:[0-9]+]], 0, [[ISEL]], 2
; CHECK-NEXT: extsw 3, [[ISEL2]]
; CHECK-NEXT: blr
}

; Check size 4
; Function Attrs: nounwind readonly
define signext i32 @test2(i32* nocapture readonly %buffer1, i32* nocapture readonly %buffer2) local_unnamed_addr #0 {
entry:
  %0 = bitcast i32* %buffer1 to i8*
  %1 = bitcast i32* %buffer2 to i8*
  %call = tail call signext i32 @memcmp(i8* %0, i8* %1, i64 4) #2
  ret i32 %call

; CHECK-LABEL: @test2
; CHECK: lwbrx [[LOAD1:[0-9]+]]
; CHECK-NEXT: lwbrx [[LOAD2:[0-9]+]]
; CHECK-NEXT: li [[LI:[0-9]+]], 1
; CHECK-NEXT: cmpld [[CMPLD:[0-9]+]], [[LOAD1]], [[LOAD2]]
; CHECK-NEXT: subf. [[SUB:[0-9]+]], [[LOAD2]], [[LOAD1]]
; CHECK-NEXT: li [[LI2:[0-9]+]], -1
; CHECK-NEXT: isel [[ISEL:[0-9]+]], [[LI2]], [[LI]], 4
; CHECK-NEXT: isel [[ISEL2:[0-9]+]], 0, [[ISEL]], 2
; CHECK-NEXT: extsw 3, [[ISEL2]]
; CHECK-NEXT: blr
}

; Check size 2
; Function Attrs: nounwind readonly
define signext i32 @test3(i32* nocapture readonly %buffer1, i32* nocapture readonly %buffer2) local_unnamed_addr #0 {
entry:
  %0 = bitcast i32* %buffer1 to i8*
  %1 = bitcast i32* %buffer2 to i8*
  %call = tail call signext i32 @memcmp(i8* %0, i8* %1, i64 2) #2
  ret i32 %call

; CHECK-LABEL: @test3
; CHECK: lhbrx [[LOAD1:[0-9]+]]
; CHECK-NEXT: lhbrx [[LOAD2:[0-9]+]]
; CHECK-NEXT: li [[LI:[0-9]+]], 1
; CHECK-NEXT: cmpld [[CMPLD:[0-9]+]], [[LOAD1]], [[LOAD2]]
; CHECK-NEXT: subf. [[SUB:[0-9]+]], [[LOAD2]], [[LOAD1]]
; CHECK-NEXT: li [[LI2:[0-9]+]], -1
; CHECK-NEXT: isel [[ISEL:[0-9]+]], [[LI2]], [[LI]], 4
; CHECK-NEXT: isel [[ISEL2:[0-9]+]], 0, [[ISEL]], 2
; CHECK-NEXT: extsw 3, [[ISEL2]]
; CHECK-NEXT: blr
}

; Check size 1
; Function Attrs: nounwind readonly
define signext i32 @test4(i32* nocapture readonly %buffer1, i32* nocapture readonly %buffer2) local_unnamed_addr #0 {
entry:
  %0 = bitcast i32* %buffer1 to i8*
  %1 = bitcast i32* %buffer2 to i8*
  %call = tail call signext i32 @memcmp(i8* %0, i8* %1, i64 1) #2
  ret i32 %call

; CHECK-LABEL: @test4
; CHECK: lbz [[LOAD1:[0-9]+]]
; CHECK-NEXT: lbz [[LOAD2:[0-9]+]]
; CHECK-NEXT: subf [[SUB:[0-9]+]], [[LOAD2]], [[LOAD1]]
; CHECK-NEXT: extsw 3, [[SUB]]
; CHECK-NEXT: blr
}

; Function Attrs: nounwind readonly
declare signext i32 @memcmp(i8*, i8*, i64) #1
