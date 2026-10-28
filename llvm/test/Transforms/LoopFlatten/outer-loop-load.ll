; RUN: opt -S -passes=loop-flatten < %s | FileCheck %s  
  
@g = global i32 7  
  
define i32 @main() {  
; CHECK-LABEL: @main( 
; CHECK-NOT: flatten.tripcount  
; CHECK: %x = load i32, ptr @g  
; CHECK: ret i32 %x.lcssa  
entry:  
  br label %outer 
  
  outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  %x = load i32, ptr @g, align 4
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner.latch ]
  store i32 0, ptr @g, align 4
  br label %inner.latch

inner.latch:
  %j.next = add i32 %j, 1
  %j.cmp = icmp ult i32 %j.next, 2
  br i1 %j.cmp, label %inner, label %outer.latch

outer.latch:
  %i.next = add i32 %i, 1
  %i.cmp = icmp ult i32 %i.next, 1
  br i1 %i.cmp, label %outer, label %exit

exit:
  ret i32 %x
}