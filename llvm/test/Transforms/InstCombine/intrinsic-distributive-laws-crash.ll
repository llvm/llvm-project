; RUN: opt -passes='default<O1>' -S %s -o /dev/null
; REQUIRES: asserts
; ModuleID = 'repro_0.ll'
source_filename = "repro.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c = external global i32
@b = external global i8

; Function Attrs: nounwind uwtable
declare void @d() #0

; Function Attrs: nounwind uwtable
define i32 @e(i32 noundef %g, i32 noundef %h, i32 noundef %i) #0 {
entry:
  %g.addr = alloca i32, align 4
  %h.addr = alloca i32, align 4
  %i.addr = alloca i32, align 4
  %f = alloca i32, align 4
  store i32 %g, ptr %g.addr, align 4
  store i32 %h, ptr %h.addr, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %g.addr, align 4
  %1 = load i32, ptr %h.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  ret i32 0

cond.false:                                       ; preds = %entry
  %2 = load i32, ptr %g.addr, align 4
  %3 = load i32, ptr %i.addr, align 4
  %cmp1 = icmp sgt i32 %2, %3
  %4 = load i32, ptr %i.addr, align 4
  %5 = load i32, ptr %g.addr, align 4
  %cond = select i1 %cmp1, i32 %4, i32 %5
  br label %cond.end

cond.true2:                                       ; No predecessors!
  br label %cond.end

cond.false3:                                      ; No predecessors!
  br label %cond.end

cond.end:                                         ; preds = %cond.false3, %cond.true2, %cond.false
  store i32 %cond, ptr %f, align 4
  %6 = load i32, ptr %f, align 4
  ret i32 %6
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define void @j() #0 {
entry:
  %k = alloca i32, align 4
  %l = alloca i32, align 4
  %m = alloca ptr, align 8
  store i32 893196426, ptr %k, align 4
  store i32 826450915, ptr %l, align 4
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  store ptr %k, ptr %m, align 8
  %0 = load i32, ptr %k, align 4
  %conv = sext i32 %0 to i64
  %shl = shl i64 %conv, 52
  %shr = ashr i64 %shl, 52
  %sub = sub nsw i64 0, %shr
  %conv1 = trunc i64 %sub to i32
  %1 = load i32, ptr %k, align 4
  %conv2 = sext i32 %1 to i64
  %shl3 = shl i64 %conv2, 32
  %shr4 = ashr i64 %shl3, 32
  %shl5 = shl i64 %shr4, 52
  %shr6 = ashr i64 %shl5, 52
  %conv7 = trunc i64 %shr6 to i32
  %add = add nsw i32 2, %conv7
  %2 = load i32, ptr %k, align 4
  %conv8 = sext i32 %2 to i64
  %shl9 = shl i64 %conv8, 32
  %shr10 = ashr i64 %shl9, 32
  %shl11 = shl i64 %shr10, 52
  %shr12 = ashr i64 %shl11, 52
  %conv13 = trunc i64 %shr12 to i32
  %sub14 = sub nsw i32 2, %conv13
  %call = call i32 @e(i32 noundef %conv1, i32 noundef %add, i32 noundef %sub14)
  store i32 %call, ptr @c, align 4
  %3 = load i32, ptr %l, align 4
  store i32 %3, ptr %k, align 4
  %4 = load ptr, ptr %m, align 8
  %5 = load i32, ptr %4, align 4
  %6 = load i8, ptr @b, align 1
  %conv15 = sext i8 %6 to i32
  %sub16 = sub nsw i32 %5, %conv15
  %sub17 = sub nsw i32 %sub16, 7
  %tobool = icmp ne i32 %sub17, 0
  %lnot = xor i1 %tobool, true
  br i1 %lnot, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  %7 = load i32, ptr null, align 4
  %tobool18 = icmp ne i32 %7, 0
  br i1 %tobool18, label %if.then, label %if.end

if.then:                                          ; preds = %do.end
  store ptr %l, ptr %m, align 8
  br label %if.end

if.end:                                           ; preds = %if.then, %do.end
  call void @d()
  %8 = load ptr, ptr %m, align 8
  %9 = load i32, ptr %8, align 4
  ret void
}

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
