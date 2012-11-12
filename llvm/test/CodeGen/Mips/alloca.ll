; RUN: llc -march=mipsel < %s | FileCheck %s

define i32 @twoalloca(i32 %size) nounwind {
entry:
; CHECK: subu  $[[T0:[0-9]+]], $sp, $[[SZ:[0-9]+]]
; CHECK: addu  $sp, $zero, $[[T0]]
; CHECK: subu  $[[T2:[0-9]+]], $sp, $[[SZ]]
; CHECK: addu  $sp, $zero, $[[T2]]
; CHECK: addu  $4, $zero, $[[T0]]
; CHECK: addu  $4, $zero, $[[T2]]
  %tmp1 = alloca i8, i32 %size, align 4
  %add.ptr = getelementptr inbounds i8* %tmp1, i32 5
  store i8 97, i8* %add.ptr, align 1
  %tmp4 = alloca i8, i32 %size, align 4
  call void @foo2(double 1.000000e+00, double 2.000000e+00, i32 3) nounwind
  %call = call i32 @foo(i8* %tmp1) nounwind
  %call7 = call i32 @foo(i8* %tmp4) nounwind
  %add = add nsw i32 %call7, %call
  ret i32 %add
}

declare void @foo2(double, double, i32)

declare i32 @foo(i8*)

@.str = private unnamed_addr constant [22 x i8] c"%d %d %d %d %d %d %d\0A\00", align 1

define i32 @alloca2(i32 %size) nounwind {
entry:
; CHECK: alloca2
; CHECK: subu  $[[T0:[0-9]+]], $sp
; CHECK: addu  $sp, $zero, $[[T0]]

  %tmp1 = alloca i8, i32 %size, align 4
  %0 = bitcast i8* %tmp1 to i32*
  %cmp = icmp sgt i32 %size, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
; CHECK: addiu $4, $[[T0]], 40

  %add.ptr = getelementptr inbounds i8* %tmp1, i32 40
  %1 = bitcast i8* %add.ptr to i32*
  call void @foo3(i32* %1) nounwind
  %arrayidx15.pre = getelementptr inbounds i8* %tmp1, i32 12
  %.pre = bitcast i8* %arrayidx15.pre to i32*
  br label %if.end

if.else:                                          ; preds = %entry
; CHECK: addiu $4, $[[T0]], 12

  %add.ptr5 = getelementptr inbounds i8* %tmp1, i32 12
  %2 = bitcast i8* %add.ptr5 to i32*
  call void @foo3(i32* %2) nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
; CHECK: lw  $5, 0($[[T0]])
; CHECK: lw  $25, %call16(printf)

  %.pre-phi = phi i32* [ %2, %if.else ], [ %.pre, %if.then ]
  %tmp7 = load i32* %0, align 4, !tbaa !0
  %arrayidx9 = getelementptr inbounds i8* %tmp1, i32 4
  %3 = bitcast i8* %arrayidx9 to i32*
  %tmp10 = load i32* %3, align 4, !tbaa !0
  %arrayidx12 = getelementptr inbounds i8* %tmp1, i32 8
  %4 = bitcast i8* %arrayidx12 to i32*
  %tmp13 = load i32* %4, align 4, !tbaa !0
  %tmp16 = load i32* %.pre-phi, align 4, !tbaa !0
  %arrayidx18 = getelementptr inbounds i8* %tmp1, i32 16
  %5 = bitcast i8* %arrayidx18 to i32*
  %tmp19 = load i32* %5, align 4, !tbaa !0
  %arrayidx21 = getelementptr inbounds i8* %tmp1, i32 20
  %6 = bitcast i8* %arrayidx21 to i32*
  %tmp22 = load i32* %6, align 4, !tbaa !0
  %arrayidx24 = getelementptr inbounds i8* %tmp1, i32 24
  %7 = bitcast i8* %arrayidx24 to i32*
  %tmp25 = load i32* %7, align 4, !tbaa !0
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i32 %tmp7, i32 %tmp10, i32 %tmp13, i32 %tmp16, i32 %tmp19, i32 %tmp22, i32 %tmp25) nounwind
  ret i32 0
}

declare void @foo3(i32*)

declare i32 @printf(i8* nocapture, ...) nounwind

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
