; RUN: llvm-profdata merge %S/Inputs/select_hash_conflict.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-instr-select=true -S | FileCheck %s
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -pgo-instr-select=true -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common dso_local local_unnamed_addr global [16 x i32] zeroinitializer, align 16
@c0 = common dso_local local_unnamed_addr global i8 0, align 1
@c1 = common dso_local local_unnamed_addr global i8 0, align 1
@c2 = common dso_local local_unnamed_addr global i8 0, align 1
@c3 = common dso_local local_unnamed_addr global i8 0, align 1
@c4 = common dso_local local_unnamed_addr global i8 0, align 1
@c5 = common dso_local local_unnamed_addr global i8 0, align 1
@c6 = common dso_local local_unnamed_addr global i8 0, align 1
@c7 = common dso_local local_unnamed_addr global i8 0, align 1
@c8 = common dso_local local_unnamed_addr global i8 0, align 1
@c9 = common dso_local local_unnamed_addr global i8 0, align 1
@c10 = common dso_local local_unnamed_addr global i8 0, align 1
@c11 = common dso_local local_unnamed_addr global i8 0, align 1
@c12 = common dso_local local_unnamed_addr global i8 0, align 1
@c13 = common dso_local local_unnamed_addr global i8 0, align 1
@c14 = common dso_local local_unnamed_addr global i8 0, align 1
@c15 = common dso_local local_unnamed_addr global i8 0, align 1

define i32 @foo(i32 %n) {
entry:
  %0 = load i8, i8* @c0, align 1
  %tobool = icmp eq i8 %0, 0
  %cond = select i1 %tobool, i32 2, i32 1
  store i32 %cond, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 0), align 16
  %1 = load i8, i8* @c1, align 1
  %tobool2 = icmp eq i8 %1, 0
  %cond3 = select i1 %tobool2, i32 2, i32 1
  store i32 %cond3, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 1), align 4
  %2 = load i8, i8* @c2, align 1
  %tobool5 = icmp eq i8 %2, 0
  %cond6 = select i1 %tobool5, i32 2, i32 1
  store i32 %cond6, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 2), align 8
  %3 = load i8, i8* @c3, align 1
  %tobool8 = icmp eq i8 %3, 0
  %cond9 = select i1 %tobool8, i32 2, i32 1
  store i32 %cond9, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 3), align 4
  %4 = load i8, i8* @c4, align 1
  %tobool11 = icmp eq i8 %4, 0
  %cond12 = select i1 %tobool11, i32 2, i32 1
  store i32 %cond12, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 4), align 16
  %5 = load i8, i8* @c5, align 1
  %tobool14 = icmp eq i8 %5, 0
  %cond15 = select i1 %tobool14, i32 2, i32 1
  store i32 %cond15, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 5), align 4
  %6 = load i8, i8* @c6, align 1
  %tobool17 = icmp eq i8 %6, 0
  %cond18 = select i1 %tobool17, i32 2, i32 1
  store i32 %cond18, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 6), align 8
  %7 = load i8, i8* @c7, align 1
  %tobool20 = icmp eq i8 %7, 0
  %cond21 = select i1 %tobool20, i32 2, i32 1
  store i32 %cond21, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 7), align 4
  %8 = load i8, i8* @c8, align 1
  %tobool23 = icmp eq i8 %8, 0
  %cond24 = select i1 %tobool23, i32 2, i32 1
  store i32 %cond24, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 8), align 16
  %9 = load i8, i8* @c9, align 1
  %tobool26 = icmp eq i8 %9, 0
  %cond27 = select i1 %tobool26, i32 2, i32 1
  store i32 %cond27, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 9), align 4
  %10 = load i8, i8* @c10, align 1
  %tobool29 = icmp eq i8 %10, 0
  %cond30 = select i1 %tobool29, i32 2, i32 1
  store i32 %cond30, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 10), align 8
  %11 = load i8, i8* @c11, align 1
  %tobool32 = icmp eq i8 %11, 0
  %cond33 = select i1 %tobool32, i32 2, i32 1
  store i32 %cond33, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 11), align 4
  %12 = load i8, i8* @c12, align 1
  %tobool35 = icmp eq i8 %12, 0
  %cond36 = select i1 %tobool35, i32 2, i32 1
  store i32 %cond36, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 12), align 16
  %13 = load i8, i8* @c13, align 1
  %tobool38 = icmp eq i8 %13, 0
  %cond39 = select i1 %tobool38, i32 2, i32 1
  store i32 %cond39, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 13), align 4
  %14 = load i8, i8* @c14, align 1
  %tobool41 = icmp eq i8 %14, 0
  %cond42 = select i1 %tobool41, i32 2, i32 1
  store i32 %cond42, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 14), align 8
  %15 = load i8, i8* @c15, align 1
  %tobool44 = icmp eq i8 %15, 0
  %cond45 = select i1 %tobool44, i32 2, i32 1
  store i32 %cond45, i32* getelementptr inbounds ([16 x i32], [16 x i32]* @a, i64 0, i64 15), align 4
  ret i32 %n
}
; CHECK-LABEL: define i32 @foo(i32 %n)
; We should skip the profile.
; CHECK-NOT: %{{.*}} = select i1 %{{.*}}, i32 2, i32 1, !prof

