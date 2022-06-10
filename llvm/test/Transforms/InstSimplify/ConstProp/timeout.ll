; NOTE: This is a timeout test for some O(something silly) constant folding behaviour. It may not be the best test. Providing it finishes, it passes.
; RUN: opt < %s -O3 -S | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8-none-eabi"

%struct.ST = type { ptr }

@global = internal global [121 x i8] zeroinitializer, align 1

define void @func() #0 {
;CHECK-LABEL: func
entry:
  %s = alloca ptr, align 4
  %j = alloca i32, align 4
  store ptr @global, ptr %s, align 4
  store i32 0, ptr %j, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %j, align 4
  %cmp = icmp slt i32 %0, 30
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load ptr, ptr %s, align 4
  %2 = bitcast ptr %1 to ptr
  %add.ptr = getelementptr inbounds i8, ptr %2, i32 4
  %3 = ptrtoint ptr %add.ptr to i32
  %4 = load ptr, ptr %s, align 4
  %5 = bitcast ptr %4 to ptr
  %add.ptr1 = getelementptr inbounds i8, ptr %5, i32 4
  %6 = ptrtoint ptr %add.ptr1 to i32
  %rem = urem i32 %6, 2
  %cmp2 = icmp eq i32 %rem, 0
  br i1 %cmp2, label %cond.true, label %cond.false

cond.true:                                        ; preds = %for.body
  br label %cond.end

cond.false:                                       ; preds = %for.body
  %7 = load ptr, ptr %s, align 4
  %8 = bitcast ptr %7 to ptr
  %add.ptr3 = getelementptr inbounds i8, ptr %8, i32 4
  %9 = ptrtoint ptr %add.ptr3 to i32
  %rem4 = urem i32 %9, 2
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 0, %cond.true ], [ %rem4, %cond.false ]
  %add = add i32 %3, %cond
  %10 = inttoptr i32 %add to ptr
  %11 = load ptr, ptr %s, align 4
  %next = getelementptr inbounds %struct.ST, ptr %11, i32 0, i32 0
  store ptr %10, ptr %next, align 4
  %12 = load ptr, ptr %s, align 4
  %next5 = getelementptr inbounds %struct.ST, ptr %12, i32 0, i32 0
  %13 = load ptr, ptr %next5, align 4
  store ptr %13, ptr %s, align 4
  br label %for.inc

for.inc:                                          ; preds = %cond.end
  %14 = load i32, ptr %j, align 4
  %inc = add nsw i32 %14, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %15 = load ptr, ptr %s, align 4
  %next6 = getelementptr inbounds %struct.ST, ptr %15, i32 0, i32 0
  store ptr null, ptr %next6, align 4
  ret void
}

