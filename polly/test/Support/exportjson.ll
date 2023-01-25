; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: opt %loadNPMPolly -polly-import-jscop-dir=%t -enable-new-pm=1 -polly -O2 -polly-export -S < %s
; RUN: FileCheck %s -input-file %t/exportjson___%entry.split---%return.jscop
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 42.0;
; }
;
define void @exportjson(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 42.0, ptr %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


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
