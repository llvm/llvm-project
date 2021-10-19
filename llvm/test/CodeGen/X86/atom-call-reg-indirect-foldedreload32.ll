; RUN: llc < %s -mtriple=i386-linux-gnu -mcpu=atom  | \
; RUN:     FileCheck --check-prefix=ATOM %s
; RUN: llc < %s -mtriple=i386-linux-gnu -mcpu=core2 | \
; RUN:     FileCheck --check-prefix=CORE2 %s
; ATOM: calll *{{%[a-z]+}}
; CORE2: calll *funcp
;
; original source code built with clang -S -emit-llvm -M32 test32.c:
;
;   int a, b, c, d, e, f, g, h, i, j;
;   extern int (*funcp)(int, int, int, int, int, int, int, int);
;   extern int sum;
;   
;   void func()
;   {
;     sum = 0;
;     for( i = a; i < b; ++i )
;     {
;       sum += (*funcp)(i, b, c, d, e, f, g, h);
;     }
;   }
;
@sum = external dso_local global i32
@a = common global i32 0, align 4
@i = common global i32 0, align 4
@b = common global i32 0, align 4
@funcp = external dso_local global i32 (i32, i32, i32, i32, i32, i32, i32, i32)*
@c = common global i32 0, align 4
@d = common global i32 0, align 4
@e = common global i32 0, align 4
@f = common global i32 0, align 4
@g = common global i32 0, align 4
@h = common global i32 0, align 4
@j = common global i32 0, align 4

define void @func() #0 {
entry:
  store i32 0, i32* @sum, align 4
  %0 = load i32, i32* @a, align 4
  store i32 %0, i32* @i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* @i, align 4
  %2 = load i32, i32* @b, align 4
  %cmp = icmp slt i32 %1, %2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32 (i32, i32, i32, i32, i32, i32, i32, i32)*, i32 (i32, i32, i32, i32, i32, i32, i32, i32)** @funcp, align 4
  %4 = load i32, i32* @i, align 4
  %5 = load i32, i32* @b, align 4
  %6 = load i32, i32* @c, align 4
  %7 = load i32, i32* @d, align 4
  %8 = load i32, i32* @e, align 4
  %9 = load i32, i32* @f, align 4
  %10 = load i32, i32* @g, align 4
  %11 = load i32, i32* @h, align 4
  %call = call i32 %3(i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11)
  %12 = load i32, i32* @sum, align 4
  %add = add nsw i32 %12, %call
  store i32 %add, i32* @sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %13 = load i32, i32* @i, align 4
  %inc = add nsw i32 %13, 1
  store i32 %inc, i32* @i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

