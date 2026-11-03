; RUN: opt -S -passes=loop-flatten < %s | FileCheck %s  
  
@g = global i32 7  
  
; %x is invariant and does not read memory, so flattening is still  
; allowed even though the inner loop stores to @g.  
define i32 @invariant_value(i32 %n) {  
; CHECK-LABEL: @invariant_value(  
; CHECK:       entry:  
; CHECK-NEXT:    %flatten.tripcount = mul i32 2, 1  
; CHECK:       outer:  
; CHECK:         %x = add i32 %n, 1  
; CHECK:       outer.latch:  
; CHECK:         %i.cmp = icmp ult i32 %i.next, %flatten.tripcount  
; CHECK:       exit:  
; CHECK:         %x.lcssa = phi i32 [ %x, %outer.latch ]  
; CHECK-NEXT:    ret i32 %x.lcssa  
entry:  
  br label %outer  
  
outer:  
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]  
  %x = add i32 %n, 1  
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
