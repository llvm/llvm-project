; RUN: opt < %s -instcombine -S | grep "ret i1 true"
; PR2993

define i1 @foo(i32 %x) {
  %1 = srem i32 %x, -1
  %2 = icmp eq i32 %1, 0
  ret i1 %2
}
