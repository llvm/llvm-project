; REQUIRES: asserts

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define internal fastcc void @compress() nounwind {
end33:
  br i1 1, label %end124, label %lor.lhs.false95

lor.lhs.false95:
  br i1 1, label %then107, label %end172

then107:
  br i1 1, label %end124, label %then113

then113:
  br label %end124

end124:
  br label %exit

end172:
  br label %exit


exit:
  unreachable


}
; CHECK-NOT: =>
; CHECK: [0] end33 => <Function Return>
; CHECK-NEXT:      [1] end33 => exit
; CHECK-NEXT:   [2] then107 => end124

; STAT: 3 region - The # of regions

; BBIT: end33, end124, exit, lor.lhs.false95, then107, then113, end172,
; BBIT: end33, end124, lor.lhs.false95, then107, then113, end172,
; BBIT: then107, then113,

; RNIT: end33 => exit, exit,
; RNIT: end33, end124, lor.lhs.false95, then107 => end124, end172,
; RNIT: then107, then113,
