; RUN: opt -licm -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "systemz-unknown"

@g_3 = external dso_local local_unnamed_addr global i32, align 4
@g_57 = external dso_local local_unnamed_addr global i8, align 2
@g_82 = external dso_local global [8 x i16], align 2
@g_107 = external dso_local local_unnamed_addr global i32, align 4

define internal fastcc void @foo1() unnamed_addr{
; CHECK-LABEL: @foo1()
entry:
  %.pre.pre = load i32, i32* @g_3, align 4
  br label %loop1

loop1:
  %tmp0 = phi i32 [ undef, %entry ], [ %var18.lcssa, %loopexit ]
  br label %preheader

preheader:
  %indvars.iv = phi i64 [ 0, %loop1 ], [ %indvars.iv.next, %loop6 ]
  %phi18 = phi i32 [ %tmp0, %loop1 ], [ 0, %loop6 ]
  %phi87 = phi i32 [ 0, %loop1 ], [ %tmp7, %loop6 ]
  %tmp1 = getelementptr inbounds [8 x i16], [8 x i16]* @g_82, i64 0, i64 %indvars.iv
  %tmp2 = load i16, i16* %tmp1, align 2
  %tmp3 = trunc i16 %tmp2 to i8
  store i8 %tmp3, i8* @g_57, align 2
  store i32 8, i32* @g_107, align 4
  %tmp4 = icmp eq i32 %.pre.pre, 0
  %spec.select = select i1 %tmp4, i32 %phi18, i32 14
  %tmp5 = trunc i64 %indvars.iv to i32
  switch i32 %spec.select, label %loopexit [
    i32 0, label %loop6
    i32 14, label %loop9
  ]

loop6:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %tmp7 = add nuw nsw i32 %phi87, 1
  %tmp8 = icmp ult i64 %indvars.iv.next, 6
  br i1 %tmp8, label %preheader, label %loop9

loop9:
  %phi8.lcssa = phi i32 [ %tmp5, %preheader ], [ %tmp7, %loop6 ]
  %tmp10 = trunc i32 %phi8.lcssa to i8
  %tmp11 = tail call i16* @func_101(i16* getelementptr inbounds ([8 x i16], [8 x i16]* @g_82, i64 0, i64 6), i16* undef, i8 zeroext %tmp10)
  unreachable

loopexit:
  %var18.lcssa = phi i32 [ %phi18, %preheader ]
  br label %loop1

}

declare dso_local i16* @func_101(i16*, i16*, i8) local_unnamed_addr

