; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s

define i32 @twoalloca(i32 %size) nounwind {
entry:
; CHECK: subu  $[[T0:[0-9]+]], $sp, $[[SZ:[0-9]+]]
; CHECK: move  $sp, $[[T0]]
; CHECK: subu  $[[T2:[0-9]+]], $sp, $[[SZ]]
; CHECK: move  $sp, $[[T2]]
; CHECK: move  $4, $[[T0]]
; CHECK: move  $4, $[[T2]]
  %tmp1 = alloca i8, i32 %size, align 4
  %add.ptr = getelementptr inbounds i8, ptr %tmp1, i32 5
  store i8 97, ptr %add.ptr, align 1
  %tmp4 = alloca i8, i32 %size, align 4
  call void @foo2(double 1.000000e+00, double 2.000000e+00, i32 3) nounwind
  %call = call i32 @foo(ptr %tmp1) nounwind
  %call7 = call i32 @foo(ptr %tmp4) nounwind
  %add = add nsw i32 %call7, %call
  ret i32 %add
}

declare void @foo2(double, double, i32)

declare i32 @foo(ptr)

@.str = private unnamed_addr constant [22 x i8] c"%d %d %d %d %d %d %d\0A\00", align 1

define i32 @alloca2(i32 %size) nounwind {
entry:
; CHECK: alloca2
; CHECK: subu  $[[T0:[0-9]+]], $sp
; CHECK: move  $sp, $[[T0]]

  %tmp1 = alloca i8, i32 %size, align 4
  %cmp = icmp sgt i32 %size, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
; CHECK: addiu $4, $[[T0]], 40

  %add.ptr = getelementptr inbounds i8, ptr %tmp1, i32 40
  call void @foo3(ptr %add.ptr) nounwind
  %arrayidx15.pre = getelementptr inbounds i8, ptr %tmp1, i32 12
  br label %if.end

if.else:                                          ; preds = %entry
; CHECK: addiu $4, $[[T0]], 12

  %add.ptr5 = getelementptr inbounds i8, ptr %tmp1, i32 12
  call void @foo3(ptr %add.ptr5) nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
; CHECK: lw  $5, 0($[[T0]])
; CHECK: lw  $25, %call16(printf)

  %arrayidx15.pre-phi = phi ptr [ %add.ptr5, %if.else ], [ %arrayidx15.pre, %if.then ]
  %tmp7 = load i32, ptr %tmp1, align 4
  %arrayidx9 = getelementptr inbounds i8, ptr %tmp1, i32 4
  %tmp10 = load i32, ptr %arrayidx9, align 4
  %arrayidx12 = getelementptr inbounds i8, ptr %tmp1, i32 8
  %tmp13 = load i32, ptr %arrayidx12, align 4
  %tmp16 = load i32, ptr %arrayidx15.pre-phi, align 4
  %arrayidx18 = getelementptr inbounds i8, ptr %tmp1, i32 16
  %tmp19 = load i32, ptr %arrayidx18, align 4
  %arrayidx21 = getelementptr inbounds i8, ptr %tmp1, i32 20
  %tmp22 = load i32, ptr %arrayidx21, align 4
  %arrayidx24 = getelementptr inbounds i8, ptr %tmp1, i32 24
  %tmp25 = load i32, ptr %arrayidx24, align 4
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %tmp7, i32 %tmp10, i32 %tmp13, i32 %tmp16, i32 %tmp19, i32 %tmp22, i32 %tmp25) nounwind
  ret i32 0
}

declare void @foo3(ptr)

declare i32 @printf(ptr nocapture, ...) nounwind
