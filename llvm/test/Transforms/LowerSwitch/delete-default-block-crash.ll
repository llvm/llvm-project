; RUN: opt < %s -passes=lowerswitch -disable-output

; This test verify -lowerswitch does not crash after deleting the default block.

declare i32 @f(i32)

define i32 @unreachable(i32 %x) {

entry:
  switch i32 %x, label %unreachable [
    i32 5, label %a
    i32 6, label %a
    i32 7, label %a
    i32 10, label %b
    i32 20, label %b
    i32 30, label %b
    i32 40, label %b
  ]
unreachable:
  unreachable
a:
  %0 = call i32 @f(i32 0)
  ret i32 %0
b:
  %1 = call i32 @f(i32 1)
  ret i32 %1
}
