; RUN: opt < %s -S | FileCheck %s

define i32 @phi_after_label(i1 %cc) {
entry:
  br i1 %cc, label %then, label %end

then:
  br label %end

end:
  %r = phi i32 [ 0, %entry ], [ 1, %then ]
  ret i32 %r
}

define void @phi_before_label(i32 %bound) {
entry:
  br label %loop

loop:
  %ctr = phi i32 [ 0, %entry ], [ %ctr.next, %loop ]
  %ctr.next = add i32 %ctr, 1
  %cc = icmp ult i32 %ctr.next, %bound
  br i1 %cc, label %loop, label %end

end:
  ret void
}

define i32 @phi_after_label_unnamed(i1 %cc) {
0:
  br i1 %cc, label %1, label %2

1:
  br label %2

2:
  %r = phi i32 [ 0, %0 ], [ 1, %1 ]
  ret i32 %r
}
