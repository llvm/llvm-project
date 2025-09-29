; REQUIRES: asserts
; RUN: not --crash opt -passes='print<access-info>' -disable-output %s

define void @unknown_dep_loopguards(ptr %a, ptr %b, ptr %c) {
entry:
  %ld.b = load i32, ptr %b
  %guard.cond = icmp slt i32 0, %ld.b
  br i1 %guard.cond, label %exit, label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %offset = add i32 %ld.b, %iv
  %gep.a.offset = getelementptr i32, ptr %a, i32 %offset
  %gep.a.offset.2 = getelementptr i32, ptr %gep.a.offset, i32 4
  %ld.a = load [4 x i32], ptr %gep.a.offset.2
  store [4 x i32] %ld.a, ptr %c
  %offset.4 = add i32 %offset, 4
  %gep.a.offset.4 = getelementptr i32, ptr %a, i32 %offset.4
  store i32 0, ptr %gep.a.offset.4
  %iv.next = add i32 %iv, 8
  %exit.cond = icmp eq i32 %iv.next, 16
  br i1 %exit.cond, label %exit, label %loop

exit:
  ret void
}
