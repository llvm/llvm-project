; RUN: opt -S --passes="ipsccp<func-spec>" -force-specialization < %s | FileCheck %s

; %if.then only becomes executable during the post-specialization solve, so nothing assigns %rem a lattice value.
; CHECK-DAG: callee.specialized.1
; CHECK-DAG: call double @fmod(double undef, double 2.000000e+00)

declare double @fmod(double, double)

define internal i32 @callee(i32 %arg) {
entry:
  %c = icmp eq i32 %arg, 1
  br i1 %c, label %ret, label %loop

loop:
  br label %loop

ret:
  ret i32 0
}

define void @caller() {
entry:
  %c1 = call i32 @callee(i32 1)
  %call = call i32 @callee(i32 0)
  %cmp = icmp sgt i32 %call, 0
  br i1 %cmp, label %if.then, label %exit

if.then:
  %rem = call double @fmod(double undef, double 2.000000e+00)
  br label %exit

exit:
  ret void
}
