; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -passes=loop-deletion -ir-dump-directory %t/logs -print-after=loop-deletion

; RUN: ls %t/logs | FileCheck %s
; CHECK: 2-{{[a-z0-9]+}}-loop-{{[a-z0-9]+}}-LoopDeletionPass-invalidated.ll

; RUN: ls %t/logs | count 1
; RUN: cat %t/logs/* | FileCheck %s --check-prefix=CHECK-CONTENTS

; CHECK-CONTENTS: ; *** IR Dump After LoopDeletionPass on bb1 (invalidated) ***
; CHECK-CONTENTS: define void @foo() {
; CHECK-CONTENTS:   br label %bb2
; CHECK-CONTENTS: bb2:                                              ; preds = %0
; CHECK-CONTENTS:   ret void
; CHECK-CONTENTS: }


define void @foo() {
  br label %bb1
bb1:
  br i1 false, label %bb1, label %bb2
bb2:
  ret void
}
