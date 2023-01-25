; RUN: llc -march=hexagon < %s | FileCheck %s

; This generates A4_addp_c, which cannot be used as a dot-new predicate
; producer (resulting in a crash).
; CHECK-NOT: p{{[0-3]+}}.new

target triple = "hexagon"

define void @ext4_group_extend() #0 {
entry:
  %es.idx.val = load i32, ptr undef, align 4
  %conv1.i = zext i32 %es.idx.val to i64
  %or.i = or i64 undef, %conv1.i
  %add20 = add i64 %or.i, undef
  %cmp21 = icmp ult i64 %add20, %or.i
  br i1 %cmp21, label %if.then23, label %if.end24

if.then23:                                        ; preds = %entry
  unreachable

if.end24:                                         ; preds = %entry
  unreachable
}

attributes #0 = { nounwind "target-cpu"="hexagonv65" }

