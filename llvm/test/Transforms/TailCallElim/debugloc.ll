; RUN: opt < %s -passes=debugify,tailcallelim -S | FileCheck %s

define i32 @foo() {
entry:
; CHECK-LABEL: entry:
; CHECK: br label %tailrecurse{{$}}

  %ret = call i32 @foo()                          ;; line 1
  ret i32 0                                       ;; line 2

; CHECK-LABEL: tailrecurse:
; CHECK: select i1 {{.+}}, !dbg ![[DbgLoc2:[0-9]+]]
; CHECK: br label %tailrecurse, !dbg ![[DbgLoc1:[0-9]+]]
}

;; Make sure tailrecurse has the call instruction's DL
; CHECK: ![[DbgLoc1]] = !DILocation(line: 1
; CHECK: ![[DbgLoc2]] = !DILocation(line: 2
