; Test that ComputeNumSignBitsForTargetNode() (SELECT_CCMASK) will help
; DAGCombiner so that it knows that %sel0 is already sign extended.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -debug-only=isel < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

%0 = type <{ ptr, i16, [6 x i8] }>
%1 = type { ptr }

define signext i16 @fun(ptr %Arg0, i16 signext %Arg1) {
entry:
  br i1 undef, label %lab0, label %lab1

lab0:
  %icmp0 = icmp eq i32 undef, 0
  %sel0 = select i1 %icmp0, i16 %Arg1, i16 1
  br label %lab1

lab1:
; CHECK: *** MachineFunction at end of ISel ***
; CHECK-LABEL: bb.2.lab1:
; CHECK-NOT:   LHR
; CHECK:       BRC
  %phi0 = phi i16 [ 2, %entry ], [ %sel0, %lab0 ]
  %sext0 = sext i16 %phi0 to i32
  br i1 undef, label %lab2, label %lab3

lab2:
  %and0 = and i32 %sext0, 8
  %icmp1 = icmp eq i32 %and0, 0
  %sel1 = select i1 %icmp1, i16 %phi0, i16 4
  ret i16 %sel1

lab3:
  ret i16 8
}

