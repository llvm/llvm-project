; RUN: llc -march=r600 -mcpu=redwood -r600-if-convert=0 < %s | FileCheck %s

; This tests for abug where the AMDILCFGStructurizer was crashing on loops
; like this:
;
; for (i = 0; i < x; i++) {
;   if (cond0) {
;     if (cond1) {
;
;     } else {
;
;     }
;     if (cond2) {
;
;     }
;   }
; }

; CHECK-LABEL: {{^}}if_inside_loop:
; CHECK: LOOP_START_DX10
; CHECK: END_LOOP
define amdgpu_kernel void @if_inside_loop(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  br label %for.body

for.body:
  %0 = phi i32 [0, %entry], [%inc, %for.inc]
  %val = phi i32 [0, %entry], [%val.for.inc, %for.inc]
  %inc = add i32 %0, 1
  %1 = icmp ult i32 10, %a
  br i1 %1, label %for.inc, label %if.then

if.then:
  %2 = icmp ne i32 0, %b
  br i1 %2, label %if.then.true, label %if.then.false

if.then.true:
  %3 = add i32 %a, %val
  br label %if

if.then.false:
  %4 = mul i32 %a, %val
  br label %if

if:
  %val.if = phi i32 [%3, %if.then.true], [%4, %if.then.false]
  %5 = icmp ne i32 0, %c
  br i1 %5, label %if.true, label %for.inc

if.true:
  %6 = add i32 %a, %val.if
  br label %for.inc

for.inc:
  %val.for.inc = phi i32 [%val, %for.body], [%val.if, %if], [%6, %if.true]
  %7 = icmp ne i32 0, %d
  br i1 %7, label %for.body, label %exit

exit:
  store i32 %val.for.inc, i32 addrspace(1)* %out
  ret void
}
