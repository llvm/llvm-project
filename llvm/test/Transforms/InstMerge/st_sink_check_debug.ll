; RUN: opt < %s -S -debugify -mldst-motion -o - | FileCheck %s

%struct.S = type { i32 }

define dso_local void @foo(%struct.S* %this, i32 %bar) {
entry:
  %this.addr = alloca %struct.S*, align 8
  %bar.addr = alloca i32, align 4
  store %struct.S* %this, %struct.S** %this.addr, align 8
  store i32 %bar, i32* %bar.addr, align 4
  %this1 = load %struct.S*, %struct.S** %this.addr, align 8
  %0 = load i32, i32* %bar.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %foo = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
  store i32 1, i32* %foo, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %foo2 = getelementptr inbounds %struct.S, %struct.S* %this1, i32 0, i32 0
  store i32 0, i32* %foo2, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; CHECK:      @foo
; CHECK:      if.end: ; preds = %if.else, %if.then
; CHECK-NEXT:   %.sink = phi {{.*}} !dbg ![[DBG:[0-9]+]]
; CHECK: ![[DBG]] = !DILocation(line: 0,
