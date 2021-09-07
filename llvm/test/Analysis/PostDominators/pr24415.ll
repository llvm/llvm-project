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
; CHECK-NEXT:   [1]  <<exit node>>
; CHECK-NEXT:     [2] %2
; CHECK-NEXT:     [2] %1
; CHECK-NEXT:       [3] %0
