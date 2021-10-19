; Note: Scaling the func entry count (using the sum of count value) so that BFI counter value is close to raw profile counter values.
; RUN: llvm-profdata merge %S/Inputs/fix_bfi.proftext -o %t.profdata
; RUN: opt -pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pgo-fix-entry-count=true < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.basket = type { %struct.arc*, i64, i64 }
%struct.arc = type { i64, %struct.node*, %struct.node*, i32, %struct.arc*, %struct.arc*, i64, i64 }
%struct.node = type { i64, i32, %struct.node*, %struct.node*, %struct.node*, %struct.node*, %struct.arc*, %struct.arc*, %struct.arc*, %struct.arc*, i64, i64, i32, i32 }

@perm = internal unnamed_addr global [351 x %struct.basket*] zeroinitializer, align 16

define dso_local void @sort_basket(i64 %min, i64 %max) {
entry:
  %add = add nsw i64 %min, %max
  %div = sdiv i64 %add, 2
  %arrayidx = getelementptr inbounds [351 x %struct.basket*], [351 x %struct.basket*]* @perm, i64 0, i64 %div
  %0 = load %struct.basket*, %struct.basket** %arrayidx, align 8
  %abs_cost = getelementptr inbounds %struct.basket, %struct.basket* %0, i64 0, i32 2
  %1 = load i64, i64* %abs_cost, align 8
  br label %do.body

do.body:
  %r.0 = phi i64 [ %max, %entry ], [ %r.2, %if.end ]
  %l.0 = phi i64 [ %min, %entry ], [ %l.2, %if.end ]
  br label %while.cond

while.cond:
  %l.1 = phi i64 [ %l.0, %do.body ], [ %inc, %while.body ]
  %arrayidx1 = getelementptr inbounds [351 x %struct.basket*], [351 x %struct.basket*]* @perm, i64 0, i64 %l.1
  %2 = load %struct.basket*, %struct.basket** %arrayidx1, align 8
  %abs_cost2 = getelementptr inbounds %struct.basket, %struct.basket* %2, i64 0, i32 2
  %3 = load i64, i64* %abs_cost2, align 8
  %cmp = icmp sgt i64 %3, %1
  br i1 %cmp, label %while.body, label %while.cond3

while.body:
  %inc = add nsw i64 %l.1, 1
  br label %while.cond

while.cond3:
  %r.1 = phi i64 [ %r.0, %while.cond ], [ %dec, %while.body7 ]
  %arrayidx4 = getelementptr inbounds [351 x %struct.basket*], [351 x %struct.basket*]* @perm, i64 0, i64 %r.1
  %4 = load %struct.basket*, %struct.basket** %arrayidx4, align 8
  %abs_cost5 = getelementptr inbounds %struct.basket, %struct.basket* %4, i64 0, i32 2
  %5 = load i64, i64* %abs_cost5, align 8
  %cmp6 = icmp sgt i64 %1, %5
  br i1 %cmp6, label %while.body7, label %while.end8

while.body7:
  %dec = add nsw i64 %r.1, -1
  br label %while.cond3

while.end8:
  %cmp9 = icmp slt i64 %l.1, %r.1
  br i1 %cmp9, label %if.then, label %if.end

if.then:
  %6 = bitcast %struct.basket** %arrayidx1 to i64*
  %7 = load i64, i64* %6, align 8
  store %struct.basket* %4, %struct.basket** %arrayidx1, align 8
  %8 = bitcast %struct.basket** %arrayidx4 to i64*
  store i64 %7, i64* %8, align 8
  br label %if.end

if.end:
  %cmp14 = icmp sgt i64 %l.1, %r.1
  %not.cmp14 = xor i1 %cmp14, true
  %9 = zext i1 %not.cmp14 to i64
  %r.2 = sub i64 %r.1, %9
  %not.cmp1457 = xor i1 %cmp14, true
  %inc16 = zext i1 %not.cmp1457 to i64
  %l.2 = add nsw i64 %l.1, %inc16
  %cmp19 = icmp sgt i64 %l.2, %r.2
  br i1 %cmp19, label %do.end, label %do.body

do.end:
  %cmp20 = icmp sgt i64 %r.2, %min
  br i1 %cmp20, label %if.then21, label %if.end22

if.then21:
  call void @sort_basket(i64 %min, i64 %r.2)
  br label %if.end22

if.end22:
  %cmp23 = icmp slt i64 %l.2, %max
  %cmp24 = icmp slt i64 %l.2, 51
  %or.cond = and i1 %cmp23, %cmp24
  br i1 %or.cond, label %if.then25, label %if.end26

if.then25:
  call void @sort_basket(i64 %l.2, i64 %max)
  br label %if.end26

if.end26:
  ret void
}

; CHECK: define dso_local void @sort_basket(i64 %min, i64 %max) #0 !prof [[ENTRY_COUNT:![0-9]+]]
; CHECK: [[ENTRY_COUNT]] = !{!"function_entry_count", i64 12949310}
