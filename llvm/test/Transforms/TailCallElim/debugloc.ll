; RUN: opt < %s -debugify -tailcallelim -S | FileCheck %s

define void @foo() {
entry:
; CHECK-LABEL: entry:
; CHECK: br label %tailrecurse, !dbg ![[DbgLoc:[0-9]+]]

  call void @foo()                            ;; line 1
  ret void

; CHECK-LABEL: tailrecurse:
; CHECK: br label %tailrecurse, !dbg ![[DbgLoc]]
}

;; Make sure tailrecurse has the call instruction's DL
; CHECK: ![[DbgLoc]] = !DILocation(line: 1
