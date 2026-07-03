; RUN: llc -mtriple=hexagon < %s | FileCheck %s
;
; Bug 6714. Use sign-extend to promote the arguments for compare
; equal/not-equal for 8- and 16-bit types with negative constants.

; CHECK: cmp.eq{{.*}}#-16
define i32 @foo1(i16 signext %q) nounwind readnone {
entry:
  %not.cmp = icmp ne i16 %q, -16
  %res.0 = zext i1 %not.cmp to i32
  ret i32 %res.0
}

; CHECK: cmp.eq{{.*}}#-14
define i32 @foo2(i16 signext %q) nounwind readnone {
entry:
  %cmp = icmp eq i16 %q, -14
  %res.0 = select i1 %cmp, i32 2, i32 0
  ret i32 %res.0
}

; CHECK: cmp.eq{{.*}}#-8
define i32 @foo3(i8 signext %r) nounwind readnone {
entry:
  %cmp = icmp eq i8 %r, -8
  %res.0 = select i1 %cmp, i32 0, i32 3
  ret i32 %res.0
}

; CHECK: cmp.eq{{.*}}#-6
define i32 @foo4(i8 signext %r) nounwind readnone {
entry:
  %cmp = icmp eq i8 %r, -6
  %res.0 = select i1 %cmp, i32 4, i32 0
  ret i32 %res.0
}

; CHECK: cmp.eq{{.*}}#-20
define i32 @foo5(i32 %s) nounwind readnone {
entry:
  %cmp = icmp eq i32 %s, -20
  %res.0 = select i1 %cmp, i32 0, i32 5
  ret i32 %res.0
}

; CHECK: cmp.eq{{.*}}#-18
define i32 @foo6(i32 %s) nounwind readnone {
entry:
  %cmp = icmp eq i32 %s, -18
  %res.0 = select i1 %cmp, i32 6, i32 0
  ret i32 %res.0
}

; CHECK: cmp.eq{{.*}}#10
define i32 @foo7(i16 signext %q) nounwind readnone {
entry:
  %cmp = icmp eq i16 %q, 10
  %res.0 = select i1 %cmp, i32 7, i32 0
  ret i32 %res.0
}

@g = external global i16

; CHECK: cmp.eq{{.*}}#-12
define i32 @foo8() nounwind readonly {
entry:
  %0 = load i16, ptr @g, align 2
  %cmp = icmp eq i16 %0, -12
  %res.0 = select i1 %cmp, i32 0, i32 8
  ret i32 %res.0
}

