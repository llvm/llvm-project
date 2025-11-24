; REQUIRES: asserts, stats
; RUN: opt -stats -passes=count-instructions < %s

define dso_local noundef i32 @add(i32 noundef %n) {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

define dso_local void @f(i32 noundef %i) {
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  store i32 0, ptr %x, align 4
  %0 = load i32, ptr %i.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 10, label %sw.bb9
    i32 1, label %sw.bb9
    i32 2, label %sw.bb10
    i32 3, label %sw.bb11
    i32 4, label %sw.bb12
  ]

sw.bb:
  %call = call noundef i32 @add(i32 noundef 9)
  store i32 %call, ptr %x, align 4
  %1 = load i32, ptr %x, align 4
  %cmp = icmp eq i32 %1, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 1, ptr %x, align 4
  br label %if.end

if.end:
  %2 = load i32, ptr %x, align 4
  %call1 = call noundef i32 @add(i32 noundef %2)
  store i32 %call1, ptr %x, align 4
  %3 = load i32, ptr %x, align 4
  %cmp2 = icmp eq i32 %3, 0
  br i1 %cmp2, label %if.then3, label %if.else

if.then3:
  store i32 1, ptr %x, align 4
  br label %if.end8

if.else:
  %4 = load i32, ptr %x, align 4
  %cmp4 = icmp eq i32 %4, 1
  br i1 %cmp4, label %if.then5, label %if.else6

if.then5:
  store i32 0, ptr %x, align 4
  br label %if.end7

if.else6:
  store i32 2, ptr %x, align 4
  br label %if.end7

if.end7:
  br label %if.end8

if.end8:
  br label %sw.epilog

sw.bb9:
  call void @h()
  br label %sw.epilog

sw.bb10:
  call void @h()
  br label %sw.epilog

sw.bb11:
  call void @j()
  br label %sw.bb12

sw.bb12:
  call void @k()
  br label %if.end15

sw.epilog:
  %5 = load i32, ptr %x, align 4
  %cmp13 = icmp eq i32 %5, 0
  br i1 %cmp13, label %if.then14, label %if.end15

if.then14:
  store i32 1, ptr %x, align 4
  br label %if.end15

if.end15:
  ret void
}

declare void @h() #2

declare void @j() #2

declare void @k() #2