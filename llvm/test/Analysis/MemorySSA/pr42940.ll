; RUN: opt -licm -verify-memoryssa  -S %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@g_77 = external dso_local global i16, align 2

; CHECK-LABEL: @f1()
define void @f1() {
entry:
  store i16 undef, ptr @g_77, align 2
  br label %loop_pre

unreachablelabel: ; No predecessors
  br label %loop_pre

loop_pre:
  br label %for.cond.header

for.cond.header:
  store i32 0, ptr undef, align 4
  br i1 undef, label %for.body, label %for.end

for.body:
  %tmp1 = load volatile i16, ptr undef, align 2
  br label %for.end

for.end:
  br i1 undef, label %func.exit, label %for.cond.header

func.exit:
  ret void
}

@g_159 = external dso_local global i32, align 4

; CHECK-LABEL: @f2()
define void @f2() {
entry:
  br label %for.header.first

for.header.first:
  br label %for.body.first

for.body.first:
  store i32 0, ptr @g_159, align 4
  br i1 undef, label %for.body.first, label %for.end.first

for.end.first:
  br i1 undef, label %lor.end, label %for.header.first

lor.end:
  br label %for.pre

unreachablelabel: ; No predecessors
  br label %for.pre

for.pre:
  br label %for.header.second

for.header.second:
  store i32 undef, ptr undef, align 4
  br label %for.header.second
}

@g_271 = external dso_local global i8, align 2
@g_427 = external dso_local unnamed_addr global [9 x i16], align 2

; CHECK-LABEL: @f3()
define  void @f3() {
entry:
  br label %for.preheader

for.preheader:
  store volatile i8 undef, ptr @g_271, align 2
  br i1 undef, label %for.preheader, label %for.end

for.end:
  br label %lbl_1058.i

unreachablelabel: ; No predecessors
  br label %lbl_1058.i

lbl_1058.i:
  br label %for.cond3.preheader.i

for.cond3.preheader.i:
  %tmp1 = load i16, ptr getelementptr inbounds ([9 x i16], ptr @g_427, i64 0, i64 2), align 2
  %conv620.i129 = zext i16 %tmp1 to i32
  %cmp621.i130 = icmp ugt i32 undef, %conv620.i129
  %conv622.i131 = zext i1 %cmp621.i130 to i32
  store i32 %conv622.i131, ptr undef, align 4
  br i1 undef, label %func.exit, label %for.cond3.preheader.i

func.exit:
  ret void
}

@g_6 = external dso_local unnamed_addr global [3 x i32], align 4
@g_244 = external dso_local global i64, align 8
@g_1164 = external dso_local global i64, align 8

; CHECK-LABEL: @f4()
define void @f4() {
entry:
  br label %for.cond8.preheader

for.cond8.preheader:
  store i32 0, ptr getelementptr inbounds ([3 x i32], ptr @g_6, i64 0, i64 2), align 4
  br i1 undef, label %if.end, label %for.cond8.preheader

if.end:
  br i1 undef, label %cleanup1270, label %for.cond504.preheader

for.cond504.preheader:
  store i64 undef, ptr @g_244, align 8
  br label %cleanup1270

for.cond559.preheader:
  store i64 undef, ptr @g_1164, align 8
  br i1 undef, label %for.cond559.preheader, label %cleanup1270

cleanup1270:
  ret void
}

@g_1504 = external dso_local local_unnamed_addr global ptr, align 8

define void @f5() {
bb:
  tail call fastcc void @f21()
  br label %bb12.outer

bb12.outer.loopexit:                              ; No predecessors!
  br label %bb12.outer

bb12.outer:                                       ; preds = %bb12.outer.loopexit, %bb
  br i1 undef, label %bb12.outer.split.us, label %bb12.preheader

bb12.preheader:                                   ; preds = %bb12.outer
  br label %bb12

bb12.outer.split.us:                              ; preds = %bb12.outer
  br label %bb16.us.us

bb16.us.us:                                       ; preds = %bb16.us.us, %bb12.outer.split.us
  br label %bb16.us.us

bb12:                                             ; preds = %bb77.1, %bb12.preheader
  br i1 undef, label %bb25.preheader, label %bb77

bb25.preheader:                                   ; preds = %bb12.1, %bb12
  br label %bb25

bb25:                                             ; preds = %l0, %bb25.preheader
  br i1 undef, label %bb62, label %bb71.thread

bb62:                                             ; preds = %bb25
  br i1 undef, label %bb92.loopexit, label %l0

l0:                                                ; preds = %bb62
  br label %bb25

bb71.thread:                                      ; preds = %bb25
  br label %bb92

bb77:                                             ; preds = %bb12
  %tmp78 = load ptr, ptr @g_1504, align 8
  %tmp79 = load volatile ptr, ptr %tmp78, align 8
  br i1 undef, label %bb91, label %bb12.1

bb91:                                             ; preds = %bb77.1, %bb77
  unreachable

bb92.loopexit:                                    ; preds = %bb62
  br label %bb92

bb92:                                             ; preds = %bb92.loopexit, %bb71.thread
  ret void

bb12.1:                                           ; preds = %bb77
  br i1 undef, label %bb25.preheader, label %bb77.1

bb77.1:                                           ; preds = %bb12.1
  br i1 undef, label %bb91, label %bb12
}

declare void @f21()
