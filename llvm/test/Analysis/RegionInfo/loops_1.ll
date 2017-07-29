; REQUIRES: asserts
; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats -disable-output < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

define internal fastcc zeroext i8 @loops_1() nounwind {
entry:
  br i1 1, label %outer , label %a

a:
  br label %body

outer:
  br label %body

body:
  br i1 1, label %land, label %if

land:
  br i1 1, label %exit, label %end

exit:
  br i1 1, label %if, label %end

if:
  br label %outer

end:
  ret i8 1
}
; CHECK-NOT: =>
; CHECK: [0] entry => <Function Return>
; CHECK-NEXT: [1] entry => end
; STAT: 2 region - The # of regions

; BBIT: entry, outer, body, land, exit, if, end, a,
; BBIT: entry, outer, body, land, exit, if, a,

; RNIT: entry => end, end,
; RNIT: entry, outer, body, land, exit, if, a,
