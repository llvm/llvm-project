;RUN: llc < %s -march=x86 | FileCheck %s

define void @foo(i32 inreg %dns) minsize {
entry:
; CHECK-LABEL: foo
; CHECK: dec
  br label %for.body

for.body:
  %i.05 = phi i16 [ %dec, %for.body ], [ 0, %entry ]
  %dec = add i16 %i.05, -1
  %conv = zext i16 %dec to i32
  %cmp = icmp slt i32 %conv, %dns
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @bar(i32 inreg %dns) minsize {
entry:
; CHECK-LABEL: bar
; CHECK: inc
  br label %for.body

for.body:
  %i.05 = phi i16 [ %inc, %for.body ], [ 0, %entry ]
  %inc = add i16 %i.05, 1
  %conv = zext i16 %inc to i32
  %cmp = icmp slt i32 %conv, %dns
  br i1 %cmp, label %for.body, label %for.end
for.end:
  ret void
}
