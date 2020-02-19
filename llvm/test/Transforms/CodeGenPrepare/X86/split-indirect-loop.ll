; RUN: opt -codegenprepare -S -mtriple=x86_64 < %s | FileCheck %s

; Test that an invalid CFG is not created by splitIndirectCriticalEdges
; transformation when the 'target' block is a loop to itself.

; CHECK: .split:
; CHECK: br label %while.body.clone
; CHECK: if.else1:
; CHECK: indirectbr
; CHECK: while.body.clone:
; CHECK: br label %.split

define void @test() {
entry:
  br label %if.else

if.else:
  br i1 undef, label %while.body, label %preheader

preheader:
  br label %if.else1

if.then:
  unreachable

while.body:
  %dest.sroa = phi i32 [ %1, %while.body ], [ undef, %if.else1 ], [ undef, %if.else ]
  %0 = inttoptr i32 %dest.sroa to i8*
  %incdec.ptr = getelementptr inbounds i8, i8* %0, i32 -1
  %1 = ptrtoint i8* %incdec.ptr to i32
  store i8 undef, i8* %incdec.ptr, align 1
  br label %while.body

if.else1:
  indirectbr i8* undef, [label %if.then, label %while.body, label %if.else, label %if.else1]
}

