;PR14492 - Tablegen incorrectly converts ARM tLDMIA_UPD pseudo to tLDMIA
;RUN: llc -mtriple=thumbv7 < %s  | FileCheck -check-prefix=EXPECTED %s
;RUN: llc -mtriple=thumbv7 < %s  | FileCheck %s

;EXPECTED-LABEL: foo:
;CHECK-LABEL: foo:
define i32 @foo(ptr %a) nounwind optsize {
entry:
  %0 = load i32, ptr %a, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %a, i32 1
  %1 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i32 2
  %2 = load i32, ptr %arrayidx2, align 4
  %add.ptr = getelementptr inbounds i32, ptr %a, i32 3
;Make sure we do not have a duplicated register in the front of the reg list
;EXPECTED:  ldm [[BASE:r[0-9]+]]!, {[[REG:r[0-9]+]], {{r[0-9]+}},
;CHECK-NOT: ldm [[BASE:r[0-9]+]]!, {[[REG:r[0-9]+]], [[REG]],
  tail call void @bar(ptr %add.ptr) nounwind optsize
  %add = add nsw i32 %1, %0
  %add3 = add nsw i32 %add, %2
  ret i32 %add3
}

declare void @bar(ptr) optsize
