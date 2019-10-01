; RUN: opt -functionattrs -attributor -attributor-disable=false -S < %s | FileCheck %s
; RUN: opt -functionattrs -attributor -attributor-disable=false -attributor-verify=true -S < %s | FileCheck %s
;
; Test cases specifically designed for the "no-return" function attribute.
; We use FIXME's to indicate problems and missing attributes.
;
; TEST 1: singleton SCC void return type
; TEST 2: singleton SCC int return type with a lot of recursive calls
; TEST 3: endless loop, no return instruction
; TEST 4: endless loop, dead return instruction
; TEST 5: all paths contain a no-return function call
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"


; TEST 1
;
; void srec0() {
;   return srec0();
; }
;
; FIXME: no-return missing
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define void @srec0()
;
define void @srec0() #0 {
entry:
  call void @srec0()
  ret void
}


; TEST 2
;
; int srec16(int a) {
;   return srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(srec16(a))))))))))))))));
; }
;
; FIXME: no-return missing
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define i32 @srec16(i32 %a)
;
define i32 @srec16(i32 %a) #0 {
entry:
  %call = call i32 @srec16(i32 %a)
  %call1 = call i32 @srec16(i32 %call)
  %call2 = call i32 @srec16(i32 %call1)
  %call3 = call i32 @srec16(i32 %call2)
  %call4 = call i32 @srec16(i32 %call3)
  %call5 = call i32 @srec16(i32 %call4)
  %call6 = call i32 @srec16(i32 %call5)
  %call7 = call i32 @srec16(i32 %call6)
  %call8 = call i32 @srec16(i32 %call7)
  %call9 = call i32 @srec16(i32 %call8)
  %call10 = call i32 @srec16(i32 %call9)
  %call11 = call i32 @srec16(i32 %call10)
  %call12 = call i32 @srec16(i32 %call11)
  %call13 = call i32 @srec16(i32 %call12)
  %call14 = call i32 @srec16(i32 %call13)
  %call15 = call i32 @srec16(i32 %call14)
  ret i32 %call15
}


; TEST 3
;
; int endless_loop(int a) {
;   while (1);
; }
;
; FIXME: no-return missing
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK: define i32 @endless_loop(i32 %a)
;
define i32 @endless_loop(i32 %a) #0 {
entry:
  br label %while.body

while.body:                                       ; preds = %entry, %while.body
  br label %while.body
}


; TEST 4
;
; int endless_loop(int a) {
;   while (1);
;   return a;
; }
;
; FIXME: no-return missing
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK: define i32 @dead_return(i32 returned %a)
;
define i32 @dead_return(i32 %a) #0 {
entry:
  br label %while.body

while.body:                                       ; preds = %entry, %while.body
  br label %while.body

return:                                           ; No predecessors!
  ret i32 %a
}


; TEST 5
;
; int multiple_noreturn_calls(int a) {
;   return a == 0 ? endless_loop(a) : srec16(a);
; }
;
; FIXME: no-return missing
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define i32 @multiple_noreturn_calls(i32 %a)
;
define i32 @multiple_noreturn_calls(i32 %a) #0 {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %call = call i32 @endless_loop(i32 %a)
  br label %cond.end

cond.false:                                       ; preds = %entry
  %call1 = call i32 @srec16(i32 %a)
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %call, %cond.true ], [ %call1, %cond.false ]
  ret i32 %cond
}

attributes #0 = { noinline nounwind uwtable }
