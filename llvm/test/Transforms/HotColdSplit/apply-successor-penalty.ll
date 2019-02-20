; REQUIRES: asserts
; RUN: opt -hotcoldsplit -debug-only=hotcoldsplit -S < %s -o /dev/null 2>&1 | FileCheck %s

declare void @sink() cold

; CHECK-LABEL: Outlining in one_non_region_successor
define void @one_non_region_successor(i32 %arg) {
entry:
  br i1 undef, label %cold1, label %exit

cold1:
  ; CHECK: Applying penalty for: 1 non-region successor
  call void @sink()
  br i1 undef, label %cold2, label %cold3

cold2:
  br i1 undef, label %cold4, label %exit

cold3:
  br i1 undef, label %cold4, label %exit

cold4:
  unreachable

exit:
  ret void
}

; CHECK-LABEL: Outlining in two_non_region_successor
define void @two_non_region_successors(i32 %arg) {
entry:
  br i1 undef, label %cold1, label %exit1

cold1:
  ; CHECK: Applying penalty for: 2 non-region successors
  call void @sink()
  br i1 undef, label %cold2, label %cold3

cold2:
  br i1 undef, label %cold4, label %exit1

cold3:
  br i1 undef, label %cold4, label %exit2

cold4:
  unreachable

exit1:
  br label %exit2

exit2:
  ret void
}
