; RUN: opt -passes=attributor -verify-each -S %s | FileCheck %s
;
; The backedge store can feed the load on later iterations.
; CHECK-LABEL: define {{.*}} @loop_counter(
; CHECK: %[[COUNT:.*]] = load i32, ptr %counter
; CHECK-NEXT: %[[NEXT:.*]] = add i32 %[[COUNT]], -1
; CHECK-NEXT: store i32 %[[NEXT]], ptr %counter
; CHECK-NEXT: %more = icmp ne i32 %[[COUNT]], 0

define i1 @loop_counter(ptr %a, i32 %n) {
entry:
  %a.addr = alloca ptr
  %counter = alloca i32
  store ptr %a, ptr %a.addr
  store i32 %n, ptr %counter
  br label %loop

loop:
  %count = load i32, ptr %counter
  %next.count = add i32 %count, -1
  store i32 %next.count, ptr %counter
  %more = icmp ne i32 %count, 0
  br i1 %more, label %body, label %return

body:
  %lhs = load ptr, ptr %a.addr
  %x = load i1, ptr %lhs
  br i1 %x, label %return, label %loop

return:
  %value = phi i1 [ true, %body ], [ false, %loop ]
  ret i1 %value
}
