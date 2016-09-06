; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global i32 0, align 4
@b = global i8 0, align 1
@c = global [4 x i8] zeroinitializer, align 1

; Just make sure we don't generate code with uses not dominated by defs.
; CHECK-LABEL: @main(
define i32 @main() {
entry:
  %a0 = load i32, i32* @a, align 4
  %cmpa = icmp slt i32 %a0, 4
  br i1 %cmpa, label %preheader, label %for.end

preheader:
  %b0 = load i8, i8* @b, align 1
  %b0sext = sext i8 %b0 to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %preheader ], [ %iv.next, %lor.false ]
  %mul = mul nsw i64 %b0sext, %iv
  %multrunc = trunc i64 %mul to i32
  %cmp = icmp eq i32 %multrunc, 0
  br i1 %cmp, label %lor.false, label %if.then

lor.false:
  %cgep = getelementptr inbounds [4 x i8], [4 x i8]* @c, i64 0, i64 %iv
  %ci = load i8, i8* %cgep, align 1
  %cisext = sext i8 %ci to i32
  %ivtrunc = trunc i64 %iv to i32
  %cmp2 = icmp eq i32 %cisext, %ivtrunc
  %iv.next = add i64 %iv, 1
  br i1 %cmp2, label %for.body, label %if.then

if.then:
  tail call void @abort()
  unreachable

for.end:
  ret i32 0
}

declare void @abort()
