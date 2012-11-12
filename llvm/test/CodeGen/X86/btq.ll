; RUN: llc < %s -march=x86-64 | FileCheck %s

declare void @bar()

define void @test1(i64 %foo) nounwind {
  %and = and i64 %foo, 4294967296
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %if.end, label %if.then

; CHECK: test1:
; CHECK: btq $32

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

define void @test2(i64 %foo) nounwind {
  %and = and i64 %foo, 2147483648
  %tobool = icmp eq i64 %and, 0
  br i1 %tobool, label %if.end, label %if.then

; CHECK: test2:
; CHECK: testl $-2147483648

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}
