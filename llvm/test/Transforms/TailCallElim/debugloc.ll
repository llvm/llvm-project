; RUN: opt < %s -passes=debugify,tailcallelim -S | FileCheck %s
; RUN: opt < %s -passes=debugify,tailcallelim -S --try-experimental-debuginfo-iterators | FileCheck %s

define void @foo() {
entry:
; CHECK-LABEL: entry:
; CHECK: br label %tailrecurse{{$}}

  call void @foo()                            ;; line 1
  ret void

; CHECK-LABEL: tailrecurse:
; CHECK: br label %tailrecurse, !dbg ![[DbgLoc:[0-9]+]]
}

;; Make sure tailrecurse has the call instruction's DL
; CHECK: ![[DbgLoc]] = !DILocation(line: 1
