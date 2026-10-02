; RUN: opt -passes="default<O3>" -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64"

; #define MAX_CNT 4
; void foo(unsigned* src, unsigned* dst, unsigned int n) {
;   for (unsigned i = 0; i < n; ++i)
;     dst[i] = src[i];
;   for (unsigned i = n; i < MAX_CNT; ++i)
;     dst[i] = -1;
; }

define void @foo(ptr noundef %src, ptr noundef %dst, i32 noundef %n) {
; CHECK-LABEL: define void @foo(
; CHECK:       call void @llvm.memset.p0.i64(ptr align 4 {{%.*}}, i8 -1, i64 range(i64 0, 17) {{%.*}}, i1 false)
entry:
  %src.addr = alloca ptr, align 8
  %dst.addr = alloca ptr, align 8
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %i3 = alloca i32, align 4
  store ptr %src, ptr %src.addr, align 8
  store ptr %dst, ptr %dst.addr, align 8
  store i32 %n, ptr %n.addr, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond
for.cond:
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %n.addr, align 4
  %cmp = icmp ult i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end
for.body:
  %2 = load ptr, ptr %src.addr, align 8
  %3 = load i32, ptr %i, align 4
  %idxprom = zext i32 %3 to i64
  %arrayidx = getelementptr inbounds nuw i32, ptr %2, i64 %idxprom
  %4 = load i32, ptr %arrayidx, align 4
  %5 = load ptr, ptr %dst.addr, align 8
  %6 = load i32, ptr %i, align 4
  %idxprom1 = zext i32 %6 to i64
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %5, i64 %idxprom1
  store i32 %4, ptr %arrayidx2, align 4
  br label %for.inc
for.inc:
  %7 = load i32, ptr %i, align 4
  %inc = add i32 %7, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond
for.end:
  %8 = load i32, ptr %n.addr, align 4
  store i32 %8, ptr %i3, align 4
  br label %for.cond4
for.cond4:
  %9 = load i32, ptr %i3, align 4
  %cmp5 = icmp ult i32 %9, 4
  br i1 %cmp5, label %for.body6, label %for.end11
for.body6:
  %10 = load ptr, ptr %dst.addr, align 8
  %11 = load i32, ptr %i3, align 4
  %idxprom7 = zext i32 %11 to i64
  %arrayidx8 = getelementptr inbounds nuw i32, ptr %10, i64 %idxprom7
  store i32 -1, ptr %arrayidx8, align 4
  br label %for.inc9
for.inc9:
  %12 = load i32, ptr %i3, align 4
  %inc10 = add i32 %12, 1
  store i32 %inc10, ptr %i3, align 4
  br label %for.cond4
for.end11:
  ret void
}
