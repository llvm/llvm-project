; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Test that the DAG ISel address tree balancing does not crash when nodes
; are replaced (RAUW'd) during balancing. This is a regression test for
; GitHub issue #64371 where use-after-poison occurred in balanceSubTree().
;
; The specific pattern that triggers the issue involves nested adds used
; in address calculations with null pointers, causing nodes to be replaced
; during the balancing process.

; CHECK-LABEL: f:
; CHECK: memb
; CHECK: memb
define void @f(i32 %LGV1, ptr %RP) {
  %G6 = getelementptr i32, ptr null, i32 %LGV1
  %B1 = add i32 %LGV1, %LGV1
  store i1 false, ptr %G6, align 1
  %B2 = add i32 %LGV1, %B1
  %G1 = getelementptr float, ptr %RP, i32 %B2
  store i1 false, ptr %G1, align 1
  ret void
}
