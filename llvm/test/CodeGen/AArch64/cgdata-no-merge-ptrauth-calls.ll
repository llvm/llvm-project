; This test checks that two similar functions, f1 and f2, are not merged when
; the difference is a direct callee and they carry the "ptrauth-calls"
; attribute, since parameterizing the callee would create an indirect call
; with no ptrauth bundle. Differences elsewhere, and callee differences in
; functions without the attribute, still merge.

; RUN: opt -mtriple=arm64e-apple-darwin -S --passes=global-merge-func %s | FileCheck %s --implicit-check-not=f1.Tgm --implicit-check-not=f2.Tgm
; RUN: llc -mtriple=arm64e-apple-darwin -enable-global-merge-func=true < %s | FileCheck %s --check-prefix=ASM --implicit-check-not=f1.Tgm --implicit-check-not=f2.Tgm

; CHECK: define internal i32 @f3.Tgm(i32 %0, ptr %1)
; CHECK: define internal i32 @f4.Tgm(i32 %0, ptr %1)
; CHECK: define internal i32 @f5.Tgm(i32 %0, ptr %1)
; CHECK: define internal i32 @f6.Tgm(i32 %0, ptr %1)

; ASM: _f3.Tgm
; ASM: _f4.Tgm
; ASM: _f5.Tgm
; ASM: _f6.Tgm

@g = external local_unnamed_addr global [0 x i32], align 4
@g1 = external global i32, align 4
@g2 = external global i32, align 4

define i32 @f1(i32 %a) "ptrauth-calls" {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = call i32 @callee1(i32 %0)
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}

define i32 @f2(i32 %a) "ptrauth-calls" {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = call i32 @callee2(i32 %0)
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}

define i32 @f3(i32 %a) "ptrauth-calls" {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = load volatile i32, ptr @g1, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}

define i32 @f4(i32 %a) "ptrauth-calls" {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = load volatile i32, ptr @g2, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}

define i32 @f5(i32 %a) {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = call i32 @callee1(i32 %0)
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 3
  ret i32 %add
}

define i32 @f6(i32 %a) {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = call i32 @callee2(i32 %0)
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 3
  ret i32 %add
}

declare i32 @callee1(i32)
declare i32 @callee2(i32)
