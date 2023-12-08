; RUN: llc < %s -mtriple=x86_64-linux-gnu -mcpu=atom  | \
; RUN:    FileCheck --check-prefix=ATOM %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu -mcpu=core2 | \
; RUN:    FileCheck --check-prefix=CORE2 %s
; ATOM: callq *{{%[a-z]+[0-9]*}}
; CORE2: callq *funcp
;
; Original source code built with clang -S -emit-llvm -m64 test64.c:
;   int a, b, c, d, e, f, g, h, i, j, k, l, m, n;
;   extern int (*funcp)(int, int, int, int, int, int,
;                       int, int, int, int, int, int,
;                       int, int);
;   extern int sum;
;   
;   void func()
;   {
;     sum = 0;
;     for( i = a; i < b; ++i )
;     {
;       sum += (*funcp)(a, i, i*2, i/b, c, d, e, f, g, h, j, k, l, n);
;     }
;   }
;   

@sum = external dso_local global i32
@a = common global i32 0, align 4
@i = common global i32 0, align 4
@b = common global i32 0, align 4
@funcp = external dso_local global ptr
@c = common global i32 0, align 4
@d = common global i32 0, align 4
@e = common global i32 0, align 4
@f = common global i32 0, align 4
@g = common global i32 0, align 4
@h = common global i32 0, align 4
@j = common global i32 0, align 4
@k = common global i32 0, align 4
@l = common global i32 0, align 4
@n = common global i32 0, align 4
@m = common global i32 0, align 4

define void @func() #0 {
entry:
  store i32 0, ptr @sum, align 4
  %0 = load i32, ptr @a, align 4
  store i32 %0, ptr @i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, ptr @i, align 4
  %2 = load i32, ptr @b, align 4
  %cmp = icmp slt i32 %1, %2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load ptr, ptr @funcp, align 8
  %4 = load i32, ptr @a, align 4
  %5 = load i32, ptr @i, align 4
  %6 = load i32, ptr @i, align 4
  %mul = mul nsw i32 %6, 2
  %7 = load i32, ptr @i, align 4
  %8 = load i32, ptr @b, align 4
  %div = sdiv i32 %7, %8
  %9 = load i32, ptr @c, align 4
  %10 = load i32, ptr @d, align 4
  %11 = load i32, ptr @e, align 4
  %12 = load i32, ptr @f, align 4
  %13 = load i32, ptr @g, align 4
  %14 = load i32, ptr @h, align 4
  %15 = load i32, ptr @j, align 4
  %16 = load i32, ptr @k, align 4
  %17 = load i32, ptr @l, align 4
  %18 = load i32, ptr @n, align 4
  %call = call i32 %3(i32 %4, i32 %5, i32 %mul, i32 %div, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i32 %15, i32 %16, i32 %17, i32 %18)
  %19 = load i32, ptr @sum, align 4
  %add = add nsw i32 %19, %call
  store i32 %add, ptr @sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %20 = load i32, ptr @i, align 4
  %inc = add nsw i32 %20, 1
  store i32 %inc, ptr @i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

