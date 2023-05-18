; REQUIRES: asserts

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define internal fastcc zeroext i8 @handle_compress() nounwind {
entry:
  br label %outer

outer:
  br label %body

body:
  br i1 1, label %body.i, label %if.end

body.i:
  br i1 1, label %end, label %if.end

if.end:
  br label %if.then64

if.then64:
  br label %outer

end:
  ret i8 1
}
; CHECK-NOT: =>
; CHECK: [0] entry => <Function Return>
; CHECK-NEXT: [1] outer => end
; STAT: 2 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: entry, outer, body, body.i, end, if.end, if.then64,
; BBIT: outer, body, body.i, if.end, if.then64,

; RNIT: entry, outer => end, end,
; RNIT: outer, body, body.i, if.end, if.then64,
