; RUN: opt < %s -passes=instcombine -S | FileCheck %s


@i = constant i32 1, align 4
@f = constant float 0x3FF19999A0000000, align 4
@cmp = common global i32 0, align 4
@resf = common global ptr null, align 8
@resi = common global ptr null, align 8

define i32 @foo() {
entry:
  br label %while.cond

while.cond:
  %res.0 = phi ptr [ null, %entry ], [ @i, %if.then ], [ @f, %if.else ]
  %0 = load i32, ptr @cmp, align 4
  %shr = ashr i32 %0, 1
  store i32 %shr, ptr @cmp, align 4
  %tobool = icmp ne i32 %shr, 0
  br i1 %tobool, label %while.body, label %while.end

while.body:
  %and = and i32 %shr, 1
  %tobool1 = icmp ne i32 %and, 0
  br i1 %tobool1, label %if.then, label %if.else

if.then:
  br label %while.cond

if.else:
  br label %while.cond

while.end:
  store ptr %res.0, ptr @resf, align 8
  store ptr %res.0, ptr @resi, align 8
  ret i32 0

; CHECK-NOT: bitcast i32
}

