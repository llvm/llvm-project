; RUN: llc < %s -mcpu=cyclone -verify-machineinstrs -aarch64-enable-ccmp | FileCheck %s
target triple = "arm64-apple-ios7.0.0"

@channelColumns = external global i64
@channelTracks = external global i64
@mazeRoute = external hidden unnamed_addr global i8*, align 8
@TOP = external global i64*
@BOT = external global i64*
@netsAssign = external global i64*

; Function from yacr2/maze.c
; The branch at the end of %if.then is driven by %cmp5 and %cmp6.
; Isel converts the and i1 into two branches, and arm64-ccmp should not convert
; it back again. %cmp6 has much higher latency than %cmp5.
; CHECK: Maze1
; CHECK: %if.then
; CHECK: cmp x{{[0-9]+}}, #2
; CHECK-NEXT: b.lo
; CHECK: %if.then
; CHECK: cmp x{{[0-9]+}}, #2
; CHECK-NEXT: b.lo
define i32 @Maze1() nounwind ssp {
entry:
  %0 = load i64, i64* @channelColumns, align 8, !tbaa !0
  %cmp90 = icmp eq i64 %0, 0
  br i1 %cmp90, label %for.end, label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %1 = phi i64 [ %0, %entry ], [ %37, %for.inc ]
  %i.092 = phi i64 [ 1, %entry ], [ %inc53, %for.inc ]
  %numLeft.091 = phi i32 [ 0, %entry ], [ %numLeft.1, %for.inc ]
  %2 = load i8*, i8** @mazeRoute, align 8, !tbaa !3
  %arrayidx = getelementptr inbounds i8, i8* %2, i64 %i.092
  %3 = load i8, i8* %arrayidx, align 1, !tbaa !1
  %tobool = icmp eq i8 %3, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %4 = load i64*, i64** @TOP, align 8, !tbaa !3
  %arrayidx1 = getelementptr inbounds i64, i64* %4, i64 %i.092
  %5 = load i64, i64* %arrayidx1, align 8, !tbaa !0
  %6 = load i64*, i64** @netsAssign, align 8, !tbaa !3
  %arrayidx2 = getelementptr inbounds i64, i64* %6, i64 %5
  %7 = load i64, i64* %arrayidx2, align 8, !tbaa !0
  %8 = load i64*, i64** @BOT, align 8, !tbaa !3
  %arrayidx3 = getelementptr inbounds i64, i64* %8, i64 %i.092
  %9 = load i64, i64* %arrayidx3, align 8, !tbaa !0
  %arrayidx4 = getelementptr inbounds i64, i64* %6, i64 %9
  %10 = load i64, i64* %arrayidx4, align 8, !tbaa !0
  %cmp5 = icmp ugt i64 %i.092, 1
  %cmp6 = icmp ugt i64 %10, 1
  %or.cond = and i1 %cmp5, %cmp6
  br i1 %or.cond, label %land.lhs.true7, label %if.else

land.lhs.true7:                                   ; preds = %if.then
  %11 = load i64, i64* @channelTracks, align 8, !tbaa !0
  %add = add i64 %11, 1
  %call = tail call fastcc i32 @Maze1Mech(i64 %i.092, i64 %add, i64 %10, i64 0, i64 %7, i32 -1, i32 -1)
  %tobool8 = icmp eq i32 %call, 0
  br i1 %tobool8, label %land.lhs.true7.if.else_crit_edge, label %if.then9

land.lhs.true7.if.else_crit_edge:                 ; preds = %land.lhs.true7
  %.pre = load i64, i64* @channelColumns, align 8, !tbaa !0
  br label %if.else

if.then9:                                         ; preds = %land.lhs.true7
  %12 = load i8*, i8** @mazeRoute, align 8, !tbaa !3
  %arrayidx10 = getelementptr inbounds i8, i8* %12, i64 %i.092
  store i8 0, i8* %arrayidx10, align 1, !tbaa !1
  %13 = load i64*, i64** @TOP, align 8, !tbaa !3
  %arrayidx11 = getelementptr inbounds i64, i64* %13, i64 %i.092
  %14 = load i64, i64* %arrayidx11, align 8, !tbaa !0
  tail call fastcc void @CleanNet(i64 %14)
  %15 = load i64*, i64** @BOT, align 8, !tbaa !3
  %arrayidx12 = getelementptr inbounds i64, i64* %15, i64 %i.092
  %16 = load i64, i64* %arrayidx12, align 8, !tbaa !0
  tail call fastcc void @CleanNet(i64 %16)
  br label %for.inc

if.else:                                          ; preds = %land.lhs.true7.if.else_crit_edge, %if.then
  %17 = phi i64 [ %.pre, %land.lhs.true7.if.else_crit_edge ], [ %1, %if.then ]
  %cmp13 = icmp ult i64 %i.092, %17
  %or.cond89 = and i1 %cmp13, %cmp6
  br i1 %or.cond89, label %land.lhs.true16, label %if.else24

land.lhs.true16:                                  ; preds = %if.else
  %18 = load i64, i64* @channelTracks, align 8, !tbaa !0
  %add17 = add i64 %18, 1
  %call18 = tail call fastcc i32 @Maze1Mech(i64 %i.092, i64 %add17, i64 %10, i64 0, i64 %7, i32 1, i32 -1)
  %tobool19 = icmp eq i32 %call18, 0
  br i1 %tobool19, label %if.else24, label %if.then20

if.then20:                                        ; preds = %land.lhs.true16
  %19 = load i8*, i8** @mazeRoute, align 8, !tbaa !3
  %arrayidx21 = getelementptr inbounds i8, i8* %19, i64 %i.092
  store i8 0, i8* %arrayidx21, align 1, !tbaa !1
  %20 = load i64*, i64** @TOP, align 8, !tbaa !3
  %arrayidx22 = getelementptr inbounds i64, i64* %20, i64 %i.092
  %21 = load i64, i64* %arrayidx22, align 8, !tbaa !0
  tail call fastcc void @CleanNet(i64 %21)
  %22 = load i64*, i64** @BOT, align 8, !tbaa !3
  %arrayidx23 = getelementptr inbounds i64, i64* %22, i64 %i.092
  %23 = load i64, i64* %arrayidx23, align 8, !tbaa !0
  tail call fastcc void @CleanNet(i64 %23)
  br label %for.inc

if.else24:                                        ; preds = %land.lhs.true16, %if.else
  br i1 %cmp5, label %land.lhs.true26, label %if.else36

land.lhs.true26:                                  ; preds = %if.else24
  %24 = load i64, i64* @channelTracks, align 8, !tbaa !0
  %cmp27 = icmp ult i64 %7, %24
  br i1 %cmp27, label %land.lhs.true28, label %if.else36

land.lhs.true28:                                  ; preds = %land.lhs.true26
  %add29 = add i64 %24, 1
  %call30 = tail call fastcc i32 @Maze1Mech(i64 %i.092, i64 0, i64 %7, i64 %add29, i64 %10, i32 -1, i32 1)
  %tobool31 = icmp eq i32 %call30, 0
  br i1 %tobool31, label %if.else36, label %if.then32

if.then32:                                        ; preds = %land.lhs.true28
  %25 = load i8*, i8** @mazeRoute, align 8, !tbaa !3
  %arrayidx33 = getelementptr inbounds i8, i8* %25, i64 %i.092
  store i8 0, i8* %arrayidx33, align 1, !tbaa !1
  %26 = load i64*, i64** @TOP, align 8, !tbaa !3
  %arrayidx34 = getelementptr inbounds i64, i64* %26, i64 %i.092
  %27 = load i64, i64* %arrayidx34, align 8, !tbaa !0
  tail call fastcc void @CleanNet(i64 %27)
  %28 = load i64*, i64** @BOT, align 8, !tbaa !3
  %arrayidx35 = getelementptr inbounds i64, i64* %28, i64 %i.092
  %29 = load i64, i64* %arrayidx35, align 8, !tbaa !0
  tail call fastcc void @CleanNet(i64 %29)
  br label %for.inc

if.else36:                                        ; preds = %land.lhs.true28, %land.lhs.true26, %if.else24
  %30 = load i64, i64* @channelColumns, align 8, !tbaa !0
  %cmp37 = icmp ult i64 %i.092, %30
  br i1 %cmp37, label %land.lhs.true38, label %if.else48

land.lhs.true38:                                  ; preds = %if.else36
  %31 = load i64, i64* @channelTracks, align 8, !tbaa !0
  %cmp39 = icmp ult i64 %7, %31
  br i1 %cmp39, label %land.lhs.true40, label %if.else48

land.lhs.true40:                                  ; preds = %land.lhs.true38
  %add41 = add i64 %31, 1
  %call42 = tail call fastcc i32 @Maze1Mech(i64 %i.092, i64 0, i64 %7, i64 %add41, i64 %10, i32 1, i32 1)
  %tobool43 = icmp eq i32 %call42, 0
  br i1 %tobool43, label %if.else48, label %if.then44

if.then44:                                        ; preds = %land.lhs.true40
  %32 = load i8*, i8** @mazeRoute, align 8, !tbaa !3
  %arrayidx45 = getelementptr inbounds i8, i8* %32, i64 %i.092
  store i8 0, i8* %arrayidx45, align 1, !tbaa !1
  %33 = load i64*, i64** @TOP, align 8, !tbaa !3
  %arrayidx46 = getelementptr inbounds i64, i64* %33, i64 %i.092
  %34 = load i64, i64* %arrayidx46, align 8, !tbaa !0
  tail call fastcc void @CleanNet(i64 %34)
  %35 = load i64*, i64** @BOT, align 8, !tbaa !3
  %arrayidx47 = getelementptr inbounds i64, i64* %35, i64 %i.092
  %36 = load i64, i64* %arrayidx47, align 8, !tbaa !0
  tail call fastcc void @CleanNet(i64 %36)
  br label %for.inc

if.else48:                                        ; preds = %land.lhs.true40, %land.lhs.true38, %if.else36
  %inc = add nsw i32 %numLeft.091, 1
  br label %for.inc

for.inc:                                          ; preds = %if.else48, %if.then44, %if.then32, %if.then20, %if.then9, %for.body
  %numLeft.1 = phi i32 [ %numLeft.091, %if.then9 ], [ %numLeft.091, %if.then20 ], [ %numLeft.091, %if.then32 ], [ %numLeft.091, %if.then44 ], [ %inc, %if.else48 ], [ %numLeft.091, %for.body ]
  %inc53 = add i64 %i.092, 1
  %37 = load i64, i64* @channelColumns, align 8, !tbaa !0
  %cmp = icmp ugt i64 %inc53, %37
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  %numLeft.0.lcssa = phi i32 [ 0, %entry ], [ %numLeft.1, %for.inc ]
  ret i32 %numLeft.0.lcssa
}

; Materializable
declare hidden fastcc i32 @Maze1Mech(i64, i64, i64, i64, i64, i32, i32) nounwind ssp

; Materializable
declare hidden fastcc void @CleanNet(i64) nounwind ssp

!0 = !{!"long", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"any pointer", !1}
