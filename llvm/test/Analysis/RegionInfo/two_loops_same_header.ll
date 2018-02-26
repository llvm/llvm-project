; REQUIRES: asserts
; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats -disable-output < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

define internal fastcc zeroext i8 @handle_compress() nounwind {
entry:
  br label %outer

outer:
  br label %body

body:
  br i1 1, label %else, label %true77

true77:
  br i1 1, label %then83, label %else

then83:
  br label %outer

else:
  br label %else106

else106:
  br i1 1, label %end, label %outer

end:
  ret i8 1
}

; CHECK-NOT: =>
; CHECK: [0] entry => <Function Return>
; CHECK-NEXT: [1] outer => end
; CHECK-NEXT:   [2] outer => else

; STAT: 3 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: entry, outer, body, else, else106, end, true77, then83,
; BBIT: outer, body, else, else106, true77, then83,
; BBIT: outer, body, true77, then83,

; RNIT: entry, outer => end, end,
; RNIT: outer => else, else, else106,
; RNIT: outer, body, true77, then83,
