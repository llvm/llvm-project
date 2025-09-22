; RUN: opt < %s -S -passes=debugify,mldst-motion -o - | FileCheck %s

%struct.S = type { i32 }

define dso_local void @foo(ptr %this, i32 %bar) {
entry:
  %this.addr = alloca ptr, align 8
  %bar.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  store i32 %bar, ptr %bar.addr, align 4
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = load i32, ptr %bar.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, ptr %this1, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 0, ptr %this1, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; CHECK:      @foo
; CHECK:      if.end: ; preds = %if.else, %if.then
; CHECK-NEXT:   %.sink = phi {{.*}} !dbg ![[DBG:[0-9]+]]
; CHECK: ![[DBG]] = !DILocation(line: 0,
