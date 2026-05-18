; RUN: llc -O2 -mtriple=hexagon -mcpu=hexagonv5 < %s
; REQUIRES: asserts
; Check for successful compilation.

@c = common global i32 0, align 4
@e = common global i32 0, align 4
@g = common global ptr null, align 4
@a = common global i32 0, align 4
@b = common global i32 0, align 4
@h = common global ptr null, align 4
@d = common global i32 0, align 4
@f = common global i32 0, align 4

define i32 @fn1(ptr nocapture readnone %p1) #0 {
entry:
  %0 = load ptr, ptr @h, align 4
  %1 = load ptr, ptr @g, align 4
  %.pre = load i32, ptr @c, align 4
  br label %for.cond

for.cond:
  %2 = phi i32 [ %10, %if.end ], [ %.pre, %entry ]
  store i32 %2, ptr @e, align 4
  %tobool5 = icmp eq i32 %2, 0
  br i1 %tobool5, label %for.end, label %for.body.lr.ph

for.body.lr.ph:
  %3 = sub i32 -5, %2
  %4 = urem i32 %3, 5
  %5 = sub i32 %3, %4
  br label %for.body

for.body:
  %add6 = phi i32 [ %2, %for.body.lr.ph ], [ %add, %for.body ]
  %6 = load i32, ptr %1, align 4
  store i32 %6, ptr @a, align 4
  %add = add nsw i32 %add6, 5
  %tobool = icmp eq i32 %add, 0
  br i1 %tobool, label %for.cond1.for.end_crit_edge, label %for.body

for.cond1.for.end_crit_edge:
  %7 = add i32 %2, 5
  %8 = add i32 %7, %5
  store i32 %8, ptr @e, align 4
  br label %for.end

for.end:
  %9 = load i32, ptr @b, align 4
  %tobool2 = icmp eq i32 %9, 0
  br i1 %tobool2, label %if.end, label %if.then

if.then:
  store i32 0, ptr %0, align 4
  %.pre7 = load i32, ptr @c, align 4
  br label %if.end

if.end:
  %10 = phi i32 [ %2, %for.end ], [ %.pre7, %if.then ]
  store i32 %10, ptr @d, align 4
  %11 = load i32, ptr @f, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, ptr @f, align 4
  br label %for.cond
}
