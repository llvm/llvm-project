; RUN: opt -passes='mem2reg,loop-simplify,lcssa,loop(loop-rotate,loop-idiom)' \
; RUN:   -mtriple=hexagon-unknown-unknown -S < %s | FileCheck %s

; C Code for generating the input file
;void foo(int n){
;  int i;
;  char a[100];
;  int b[100];
;
;  for(i=n; i> 8; i--)
;    a[i] = a[i-8]; // should promote to memmove
;  for(i=n; i> 1; i--)
;    b[i] = b[i-2]; // should promote to memmove
;
;  for(i=n; i> 0; i--)
;    a[i] = a[i-1]; // unaligned load or store, should not promote to memmove
;
;}

; CHECK-LABEL: @foo(
; CHECK:       call {{.*}}memmove
; CHECK:       call {{.*}}memmove
; CHECK-NOT:   call {{.*}}memmove
; CHECK:       getelementptr {{.*}} %a
; CHECK:       load i8
; CHECK:       getelementptr {{.*}} %a
; CHECK:       store i8
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-unknown-elf"

define dso_local void @foo(i32 %n)  {
entry:
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %a = alloca [100 x i8], align 8
  %b = alloca [100 x i32], align 8
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32, i32* %n.addr, align 4
  store i32 %0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4
  %cmp = icmp sgt i32 %1, 8
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4
  %sub = sub nsw i32 %2, 8
  %arrayidx = getelementptr inbounds [100 x i8], [100 x i8]* %a, i32 0, i32 %sub
  %3 = load i8, i8* %arrayidx, align 1
  %4 = load i32, i32* %i, align 4
  %arrayidx1 = getelementptr inbounds [100 x i8], [100 x i8]* %a, i32 0, i32 %4
  store i8 %3, i8* %arrayidx1, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %dec = add nsw i32 %5, -1
  store i32 %dec, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %n.addr, align 4
  store i32 %6, i32* %i, align 4
  br label %for.cond2
for.cond2:                                        ; preds = %for.inc8, %for.end
  %7 = load i32, i32* %i, align 4
  %cmp3 = icmp sgt i32 %7, 1
  br i1 %cmp3, label %for.body4, label %for.end10

for.body4:                                        ; preds = %for.cond2
  %8 = load i32, i32* %i, align 4
  %sub5 = sub nsw i32 %8, 2
  %arrayidx6 = getelementptr inbounds [100 x i32], [100 x i32]* %b, i32 0, i32 %sub5
  %9 = load i32, i32* %arrayidx6, align 4
  %10 = load i32, i32* %i, align 4
  %arrayidx7 = getelementptr inbounds [100 x i32], [100 x i32]* %b, i32 0, i32 %10
  store i32 %9, i32* %arrayidx7, align 4
  br label %for.inc8

for.inc8:                                         ; preds = %for.body4
  %11 = load i32, i32* %i, align 4
  %dec9 = add nsw i32 %11, -1
  store i32 %dec9, i32* %i, align 4
  br label %for.cond2

for.end10:                                        ; preds = %for.cond2
  %12 = load i32, i32* %n.addr, align 4
  store i32 %12, i32* %i, align 4
  br label %for.cond11

for.cond11:                                       ; preds = %for.inc17, %for.end10
  %13 = load i32, i32* %i, align 4
  %cmp12 = icmp sgt i32 %13, 0
  br i1 %cmp12, label %for.body13, label %for.end19

for.body13:                                       ; preds = %for.cond11
  %14 = load i32, i32* %i, align 4
  %sub14 = sub nsw i32 %14, 1
  %arrayidx15 = getelementptr inbounds [100 x i8], [100 x i8]* %a, i32 0, i32 %sub14
  %15 = load i8, i8* %arrayidx15, align 1
  %16 = load i32, i32* %i, align 4
  %arrayidx16 = getelementptr inbounds [100 x i8], [100 x i8]* %a, i32 0, i32 %16
  store i8 %15, i8* %arrayidx16, align 1
  br label %for.inc17

for.inc17:                                        ; preds = %for.body13
  %17 = load i32, i32* %i, align 4
  %dec18 = add nsw i32 %17, -1
  store i32 %dec18, i32* %i, align 4
  br label %for.cond11

for.end19:                                        ; preds = %for.cond11
  ret void
}
