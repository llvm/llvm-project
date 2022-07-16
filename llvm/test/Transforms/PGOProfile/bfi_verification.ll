; Note: Verify bfi counter after loading the profile.
; RUN: llvm-profdata merge %S/Inputs/bfi_verification.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pgo-verify-bfi-ratio=2 -pgo-verify-bfi=true -pgo-fix-entry-count=false -pass-remarks-analysis=pgo 2>&1 | FileCheck %s --check-prefix=THRESHOLD-CHECK
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pgo-verify-hot-bfi=true -pgo-fix-entry-count=false -pass-remarks-analysis=pgo 2>&1 | FileCheck %s --check-prefix=HOTONLY-CHECK

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.basket = type { ptr, i64, i64 }
%struct.arc = type { i64, ptr, ptr, i32, ptr, ptr, i64, i64 }
%struct.node = type { i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, i32, i32 }

@perm = internal unnamed_addr global [351 x ptr] zeroinitializer, align 16

define dso_local void @sort_basket(i64 %min, i64 %max) {
entry:
  %add = add nsw i64 %min, %max
  %div = sdiv i64 %add, 2
  %arrayidx = getelementptr inbounds [351 x ptr], ptr @perm, i64 0, i64 %div
  %0 = load ptr, ptr %arrayidx, align 8
  %abs_cost = getelementptr inbounds %struct.basket, ptr %0, i64 0, i32 2
  %1 = load i64, ptr %abs_cost, align 8
  br label %do.body

do.body:
  %r.0 = phi i64 [ %max, %entry ], [ %r.2, %if.end ]
  %l.0 = phi i64 [ %min, %entry ], [ %l.2, %if.end ]
  br label %while.cond

while.cond:
  %l.1 = phi i64 [ %l.0, %do.body ], [ %inc, %while.body ]
  %arrayidx1 = getelementptr inbounds [351 x ptr], ptr @perm, i64 0, i64 %l.1
  %2 = load ptr, ptr %arrayidx1, align 8
  %abs_cost2 = getelementptr inbounds %struct.basket, ptr %2, i64 0, i32 2
  %3 = load i64, ptr %abs_cost2, align 8
  %cmp = icmp sgt i64 %3, %1
  br i1 %cmp, label %while.body, label %while.cond3

while.body:
  %inc = add nsw i64 %l.1, 1
  br label %while.cond

while.cond3:
  %r.1 = phi i64 [ %r.0, %while.cond ], [ %dec, %while.body7 ]
  %arrayidx4 = getelementptr inbounds [351 x ptr], ptr @perm, i64 0, i64 %r.1
  %4 = load ptr, ptr %arrayidx4, align 8
  %abs_cost5 = getelementptr inbounds %struct.basket, ptr %4, i64 0, i32 2
  %5 = load i64, ptr %abs_cost5, align 8
  %cmp6 = icmp sgt i64 %1, %5
  br i1 %cmp6, label %while.body7, label %while.end8

while.body7:
  %dec = add nsw i64 %r.1, -1
  br label %while.cond3

while.end8:
  %cmp9 = icmp slt i64 %l.1, %r.1
  br i1 %cmp9, label %if.then, label %if.end

if.then:
  %6 = load i64, ptr %arrayidx1, align 8
  store ptr %4, ptr %arrayidx1, align 8
  store i64 %6, ptr %arrayidx4, align 8
  br label %if.end

if.end:
  %cmp14 = icmp sgt i64 %l.1, %r.1
  %not.cmp14 = xor i1 %cmp14, true
  %7 = zext i1 %not.cmp14 to i64
  %r.2 = sub i64 %r.1, %7
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
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB do.body Count=39637749 BFI_Count=40801304
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB while.cond Count=80655628 BFI_Count=83956530
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB while.body Count=41017879 BFI_Count=42370585
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB while.cond3 Count=71254487 BFI_Count=73756204
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB while.body7 Count=31616738 BFI_Count=32954900
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB while.end8 Count=39637749 BFI_Count=40801304
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB if.then Count=32743703 BFI_Count=33739540
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB if.end Count=39637749 BFI_Count=40801304
; THRESHOLD-CHECK: remark: <unknown>:0:0: BB if.then25 Count=6013544 BFI_Count=6277124
; THRESHOLD-CHECK: remark: <unknown>:0:0: In Func sort_basket: Num_of_BB=14, Num_of_non_zerovalue_BB=14, Num_of_mis_matching_BB=9
; HOTONLY-CHECK: remark: <unknown>:0:0: BB if.then25 Count=6013544 BFI_Count=6277124 (raw-Cold to BFI-Hot)
; HOTONLY-CHECK: remark: <unknown>:0:0: In Func sort_basket: Num_of_BB=14, Num_of_non_zerovalue_BB=14, Num_of_mis_matching_BB=1
