; RUN: opt < %s -postdomtree -analyze | FileCheck %s
; RUN: opt < %s -passes='print<postdomtree>' 2>&1 | FileCheck %s

; Function Attrs: nounwind ssp uwtable
define void @foo() {
  br label %1

; <label>:1                                       ; preds = %0, %1
  br label %1
                                                  ; No predecessors!
  ret void
}

; CHECK: Inorder PostDominator Tree: 
; CHECK-NEXT:   [1]  <<exit node>> {0,7}
; CHECK-NEXT:     [2] %2 {1,2}
; CHECK-NEXT:     [2] %1 {3,6}
; CHECK-NEXT:       [3] %0 {4,5}
