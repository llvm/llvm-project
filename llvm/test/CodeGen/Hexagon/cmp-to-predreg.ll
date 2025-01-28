; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check that we generate compare to predicate register.

define i32 @compare1(i32 %a, i32 %b) nounwind {
; CHECK: p{{[0-3]}} = {{!?}}cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
entry:
  %cmp = icmp ne i32 %a, %b
  %add = add nsw i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %add.sub = select i1 %cmp, i32 %add, i32 %sub
  ret i32 %add.sub
}

define i32 @compare2(i32 %a) nounwind {
; CHECK: p{{[0-3]}} = {{!?}}cmp.eq(r{{[0-9]+}},#10)
entry:
  %cmp = icmp ne i32 %a, 10
  %add = add nsw i32 %a, 10
  %sub = sub nsw i32 %a, 10
  %add.sub = select i1 %cmp, i32 %add, i32 %sub
  ret i32 %add.sub
}

define i32 @compare3(i32 %a, i32 %b) nounwind {
; CHECK: p{{[0-3]}} = cmp.gt(r{{[0-9]+}},r{{[0-9]+}})
entry:
  %cmp = icmp sgt i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %add = add nsw i32 %a, %b
  %sub.add = select i1 %cmp, i32 %sub, i32 %add
  ret i32 %sub.add
}

define i32 @compare4(i32 %a) nounwind {
; CHECK: p{{[0-3]}} = cmp.gt(r{{[0-9]+}},#10)
entry:
  %cmp = icmp sgt i32 %a, 10
  %sub = sub nsw i32 %a, 10
  %add = add nsw i32 %a, 10
  %sub.add = select i1 %cmp, i32 %sub, i32 %add
  ret i32 %sub.add
}

