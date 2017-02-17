; Check that a callee-saved register will be saved correctly if
; the predicate-to-GPR spilling code uses it.
;
; RUN: llc -march=hexagon < %s | FileCheck %s
;
; We expect to spill p0 into a general-purpose register and keep it there,
; without adding an extra spill of that register.
;
; CHECK: PredSpill:
; CHECK-DAG: r{{[0-9]+}} = p0
; CHECK-DAG: p0 = r{{[0-9]+}}
; CHECK-NOT: = memw(r29
;

define void @PredSpill() {
entry:
  br i1 undef, label %if.then, label %if.else.14

if.then:                                          ; preds = %entry
  br i1 undef, label %if.end.57, label %if.else

if.else:                                          ; preds = %if.then
  unreachable

if.else.14:                                       ; preds = %entry
  br i1 undef, label %if.then.17, label %if.end.57

if.then.17:                                       ; preds = %if.else.14
  br i1 undef, label %if.end.57, label %if.then.20

if.then.20:                                       ; preds = %if.then.17
  %call21 = tail call i32 @myfun()
  %tobool22 = icmp eq i32 %call21, 0
  %0 = tail call i32 @myfun()
  br i1 %tobool22, label %if.else.42, label %if.then.23

if.then.23:                                       ; preds = %if.then.20
  unreachable

if.else.42:                                       ; preds = %if.then.20
  ret void

if.end.57:                                        ; preds = %if.then.17, %if.else.14, %if.then
  ret void
}

declare i32 @myfun()

