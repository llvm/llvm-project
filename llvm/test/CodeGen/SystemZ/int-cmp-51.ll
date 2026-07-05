; Check that modelling of CC/CCRegs does not stop MachineCSE from
; removing a compare.  MachineCSE will not extend a live range of an
; allocatable or reserved phys reg.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @bar(i8)

; Check the low end of the CH range.
define void @f1(i32 %lhs) {
; CHECK-LABEL: %bb.1:
; CHECK-NOT: cijlh %r0, 1, .LBB0_3

entry:
  %and188 = and i32 %lhs, 255
  %cmp189 = icmp ult i32 %and188, 2
  br i1 %cmp189, label %if.then.191, label %if.else.201

if.then.191:
  %cmp194 = icmp eq i32 %and188, 1
  br i1 %cmp194, label %if.then.196, label %if.else.198

if.then.196:
  call void @bar(i8 1);
  br label %if.else.201

if.else.198:
  call void @bar(i8 0);
  br label %if.else.201

if.else.201:
  ret void
}

