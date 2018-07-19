; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate compare to general register.

define i32 @compare1(i32 %a) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}cmp.eq(r{{[0-9]+}},{{ *}}#120)
entry:
  %cmp = icmp eq i32 %a, 120
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @compare2(i32 %a) nounwind readnone {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}!cmp.eq(r{{[0-9]+}},{{ *}}#120)
entry:
  %cmp = icmp ne i32 %a, 120
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @compare3(i32 %a, i32 %b) nounwind readnone {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}cmp.eq(r{{[0-9]+}},{{ *}}r{{[0-9]+}})
entry:
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @compare4(i32 %a, i32 %b) nounwind readnone {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}!cmp.eq(r{{[0-9]+}},{{ *}}r{{[0-9]+}})
entry:
  %cmp = icmp ne i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
