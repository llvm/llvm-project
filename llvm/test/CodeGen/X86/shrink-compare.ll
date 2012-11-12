; RUN: llc < %s -march=x86-64 | FileCheck %s

declare void @bar()

define void @test1(i32* nocapture %X) nounwind {
entry:
  %tmp1 = load i32* %X, align 4
  %and = and i32 %tmp1, 255
  %cmp = icmp eq i32 %and, 47
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
; CHECK: test1:
; CHECK: cmpb $47, (%{{rdi|rcx}})
}

define void @test2(i32 %X) nounwind {
entry:
  %and = and i32 %X, 255
  %cmp = icmp eq i32 %and, 47
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
; CHECK: test2:
; CHECK: cmpb $47, %{{dil|cl}}
}
