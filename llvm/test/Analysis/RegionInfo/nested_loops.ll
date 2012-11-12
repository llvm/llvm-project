; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s

; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define internal fastcc zeroext i8 @handle_compress() nounwind {
entry:
  br label %outer

outer:
  br label %body

body:
  br i1 1, label %exit172, label %end

exit172:
  br i1 1, label %end, label %outer

end:
  ret i8 1
}
; CHECK-NOT: =>
; CHECK: [0] entry => <Function Return>
; CHECK-NEXT: [1] outer => end

; STAT: 2 region - The # of regions

; BBIT: entry, outer, body, exit172, end,
; BBIT: outer, body, exit172,

; RNIT: entry, outer => end, end,
; RNIT: outer, body, exit172,
