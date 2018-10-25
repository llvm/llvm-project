; RUN: opt < %s -loop-rotate -S | FileCheck %s
; RUN: opt < %s -loop-rotate -enable-mssa-loop-dependency=true -verify-memoryssa -S | FileCheck %s

@a = external global i8, align 4
@tmp = global i8* @a

define void @f() {
; CHECK-LABEL: define void @f(
; CHECK: getelementptr i8, i8* @a, i32 0
entry:
  br label %for.preheader

for.preheader:
  br i1 undef, label %if.then8, label %for.body

for.body:
  br i1 undef, label %if.end, label %if.then8

if.end:
  %arrayidx = getelementptr i8, i8* @a, i32 0
  br label %for.preheader

if.then8:
  unreachable
}
