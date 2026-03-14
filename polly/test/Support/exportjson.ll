; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: opt %loadNPMPolly -polly-import-jscop-dir=%t '-passes=polly-custom<export-jscop>' -disable-output < %s
; RUN: FileCheck %s -input-file %t/exportjson___%entry.split---%return.jscop
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 42.0;
; }
;
define void @exportjson(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %entry.split

entry.split:
  %j.cmp1 = icmp sgt i32 %n, 0
  br i1 %j.cmp1, label %body.lr.ph, label %return

body.lr.ph:
  store double 4.200000e+01, ptr %A, align 8
  br label %return

return:
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) }


; CHECK:      {
; CHECK-NEXT:    "arrays": [
; CHECK-NEXT:       {
; CHECK-NEXT:          "name": "MemRef_A",
; CHECK-NEXT:          "sizes": [
; CHECK-NEXT:             "*"
; CHECK-NEXT:          ],
; CHECK-NEXT:          "type": "double"
; CHECK-NEXT:       }
; CHECK-NEXT:    ],
; CHECK-NEXT:    "context": "[n] -> {  : -2147483648 <= n <= 2147483647 }",
; CHECK-NEXT:    "name": "%entry.split---%return",
; CHECK-NEXT:    "statements": [
; CHECK-NEXT:       {
; CHECK-NEXT:          "accesses": [
; CHECK-NEXT:             {
; CHECK-NEXT:                "kind": "write",
; CHECK-NEXT:                "relation": "[n] -> { Stmt_body_lr_ph[] -> MemRef_A[0] }"
; CHECK-NEXT:             }
; CHECK-NEXT:          ],
; CHECK-NEXT:          "domain": "[n] -> { Stmt_body_lr_ph[] : n > 0 }",
; CHECK-NEXT:          "name": "Stmt_body_lr_ph",
; CHECK-NEXT:          "schedule": "[n] -> { Stmt_body_lr_ph[] -> [] }"
; CHECK-NEXT:       }
; CHECK-NEXT:    ]
; CHECK-NEXT: }
