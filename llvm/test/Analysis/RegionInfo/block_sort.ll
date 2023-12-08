; REQUIRES: asserts

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @BZ2_blockSort() nounwind {
start:
  br label %while

while:
  br label %while.body134.i.i

while.body134.i.i:
  br i1 1, label %end, label %w

w:
  br label %if.end140.i.i

if.end140.i.i:
  br i1 1, label %while.end186.i.i, label %if.end183.i.i

if.end183.i.i:
  br label %while.body134.i.i

while.end186.i.i:
  br label %while

end:
  ret void
}
; CHECK-NOT: =>
; CHECK: [0] start => <Function Return>
; CHECK: [1] while => end

; STAT: 2 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: start, while, while.body134.i.i, end, w, if.end140.i.i, while.end186.i.i, if.end183.i.i,
; BBIT: while, while.body134.i.i, w, if.end140.i.i, while.end186.i.i, if.end183.i.i,

; RNIT: start, while => end, end,
; RNIT: while, while.body134.i.i, w, if.end140.i.i, while.end186.i.i, if.end183.i.i,
