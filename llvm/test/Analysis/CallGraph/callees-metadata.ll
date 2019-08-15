; RUN: opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s

; CHECK:      Call graph node <<null function>><<{{.+}}>> #uses=0
; CHECK-DAG:    CS<0x0> calls function 'main'
; CHECK-DAG:    CS<0x0> calls function 'add'
; CHECK-DAG:    CS<0x0> calls function 'sub'
;
; CHECK:      Call graph node for function: 'add'<<{{.+}}>> #uses=2
;
; CHECK:      Call graph node for function: 'main'<<{{.+}}>> #uses=1
; CHECK-NEXT:   CS<{{.+}}> calls <<null function>><<[[CALLEES:.+]]>>
;
; CHECK:      Call graph node for function: 'sub'<<{{.+}}>> #uses=2
;
; CHECK:      Call graph node <<null function>><<[[CALLEES]]>> #uses=1
; CHECK-DAG:    CS<0x0> calls function 'add'
; CHECK-DAG:    CS<0x0> calls function 'sub'

define i64 @main(i64 %x, i64 %y, i64 (i64, i64)* %binop) {
  %tmp0 = call i64 %binop(i64 %x, i64 %y), !callees !0
  ret i64 %tmp0
}

define i64 @add(i64 %x, i64 %y) {
  %tmp0 = add i64 %x, %y
  ret i64 %tmp0
}

define i64 @sub(i64 %x, i64 %y) {
  %tmp0 = sub i64 %x, %y
  ret i64 %tmp0
}

!0 = !{i64 (i64, i64)* @add, i64 (i64, i64)* @sub}
