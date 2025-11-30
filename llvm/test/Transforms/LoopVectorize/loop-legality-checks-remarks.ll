; RUN: opt < %s -passes=loop-vectorize -pass-remarks=loop-vectorize -debug -disable-output 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure LV legal bails out when the loop doesn't have a legal pre-header.
; CHECK-LABEL: 'not_exist_preheader'
; CHECK: LV: Not vectorizing: Loop doesn't have a legal pre-header.
define void @not_exist_preheader(ptr %currMB, i1 %arg, ptr %arg2) nounwind uwtable {

start.exit:
  indirectbr ptr %arg2, [label %0, label %for.bodyprime]

0:
  unreachable

for.bodyprime:
  %i.057375 = phi i32 [0, %start.exit], [%1, %for.bodyprime]
  %arrayidx8prime = getelementptr inbounds i32, ptr %currMB, i32 %i.057375
  store i32 0, ptr %arrayidx8prime, align 4
  %1 = add i32 %i.057375, 1
  %cmp5prime = icmp slt i32 %1, 4
  br i1 %cmp5prime, label %for.bodyprime, label %exit

exit:
  ret void
}
