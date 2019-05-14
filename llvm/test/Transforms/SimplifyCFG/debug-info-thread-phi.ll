; RUN: opt %s -debugify -simplifycfg -S | FileCheck %s
; Tests Bug 37966

define void @bar(i32 %aa) {
; CHECK-LABEL: @bar(
; CHECK: if.end.1.critedge:
; CHECK: br label %if.end.1, !dbg ![[DBG:[0-9]+]]
entry:
  %aa.addr = alloca i32, align 4
  %bb = alloca i32, align 4
  store i32 %aa, i32* %aa.addr, align 4
  store i32 0, i32* %bb, align 4
  %tobool = icmp ne i32 %aa, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  store i32 1, i32* %bb, align 4
  br i1 %tobool, label %if.then.1, label %if.end.1 ; "line 10" to -debugify

if.then.1:                                        ; preds = %if.end
  call void @foo()
  br label %if.end.1

if.end.1:                                         ; preds = %if.then.1, %if.end
  store i32 2, i32* %bb, align 4
  br label %for.end

for.end:                                          ; preds = %if.end.1
  ret void
}

declare void @foo()

; CHECK: ![[DBG]] = !DILocation(line: 10,
