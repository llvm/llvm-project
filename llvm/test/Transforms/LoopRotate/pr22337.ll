; RUN: opt < %s -passes=loop-rotate -verify-memoryssa -S | FileCheck %s

@a = external global i8, align 4
@tmp = global ptr @a

define void @f() {
; CHECK-LABEL: define void @f(
; CHECK: getelementptr i8, ptr @a, i32 1
entry:
  br label %for.preheader

for.preheader:
  br i1 undef, label %if.then8, label %for.body

for.body:
  br i1 undef, label %if.end, label %if.then8

if.end:
  %arrayidx = getelementptr i8, ptr @a, i32 1
  br label %for.preheader

if.then8:
  unreachable
}
