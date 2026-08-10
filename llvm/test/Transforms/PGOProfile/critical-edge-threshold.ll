; Test the critical edge threahold
; RUN: opt < %s -passes=pgo-instr-gen -pgo-critical-edge-threshold=1 -pgo-instrument-entry=true -S | FileCheck %s

@sum = dso_local global i32 0, align 4

define void @foo(i32 %a, i32 %b) {
entry:
  %tobool.not = icmp eq i32 %a, 0
  br i1 %tobool.not, label %if.end4, label %if.then

if.then:
  %0 = load i32, ptr @sum, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @sum, align 4
  %tobool1.not = icmp eq i32 %b, 0
  br i1 %tobool1.not, label %if.end4, label %if.then2

if.then2:
  %inc3 = add nsw i32 %0, 2
  store i32 %inc3, ptr @sum, align 4
  br label %if.end4

if.end4:
  ret void
}

; CHECK-NOT: call void @llvm.instrprof.increment(ptr @__profn_foo
