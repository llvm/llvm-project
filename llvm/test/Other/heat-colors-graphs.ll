; RUN: opt < %s -dot-cfg -cfg-heat-colors -cfg-dot-filename-prefix=%t 2>/dev/null
; RUN: FileCheck %s -input-file=%t.f.dot --check-prefixes=CHECK-CFG,CHECK-BOTH
; RUN: opt %s -dot-callgraph -callgraph-heat-colors -callgraph-dot-filename-prefix=%t 2>/dev/null
; RUN: FileCheck %s -input-file=%t.callgraph.dot --check-prefix=CHECK-BOTH

; CHECK-BOTH: color="#{{[(a-z)(0-9)]+}}", style={{[a-z]+}}, fillcolor="#{{[(a-z)(0-9)]+}}"
; CHECK-CFG: color="#{{[(a-z)(0-9)]+}}", style={{[a-z]+}}, fillcolor="#{{[(a-z)(0-9)]+}}"
; CHECK-CFG: color="#{{[(a-z)(0-9)]+}}", style={{[a-z]+}}, fillcolor="#{{[(a-z)(0-9)]+}}"

define void @f(i32) {
entry:
  %check = icmp sgt i32 %0, 0
  br i1 %check, label %if, label %exit, !prof !0

if:                     ; preds = %entry
  br label %exit
exit:                   ; preds = %entry, %if
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 200}
