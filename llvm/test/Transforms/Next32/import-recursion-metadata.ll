; RUN: opt %s -S -passes=next-silicon-import-recursion | FileCheck %s

; CHECK-LABEL: @func_with_location()
; CHECK-SAME: #[[ATTR_LIST_0:[0-9+]]]
define void @func_with_location() "ns-location"="grid" {
  ret void
}

; CHECK-LABEL: @func_with_loop_location
; CHECK-SAME: #[[ATTR_LIST_1:[0-9+]]]
define internal void @func_with_loop_location(i1 %cond) {
  br label %loop

loop:
  br i1 %cond, label %loop, label %exit, !llvm.loop !0

exit:
  ret void
}

; CHECK: attributes #[[ATTR_LIST_0]] = { "ns-import-recursion" "ns-location"="grid" }
; CHECK: attributes #[[ATTR_LIST_1]] = { "ns-import-recursion" }

!0 = distinct !{!0, !1}
!1 = !{!"ns.loop.location", !"grid"}
