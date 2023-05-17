; RUN: llc -march=hexagon -mcpu=hexagonv5 -O2 -disable-block-placement=0 < %s | FileCheck %s

; Test that there is no redundant register assignment in the hardware loop
; preheader.

; CHECK-NOT: r{{.*}} = #5

@g = external global i32

define void @foo() #0 {
entry:
  br i1 undef, label %if.end38, label %for.body

for.body:
  %loopIdx.051 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  store i32 1, ptr @g, align 4
  %inc = add i32 %loopIdx.051, 1
  %cmp9 = icmp ult i32 %inc, 5
  br i1 %cmp9, label %for.body, label %if.end38

if.end38:
  ret void
}
