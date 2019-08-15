; RUN: opt < %s -passes=print-lcg -disable-output 2>&1 | FileCheck %s

; CHECK:      Edges in function: main
; CHECK-DAG:    ref  -> add
; CHECK-DAG:    ref  -> sub
;
; CHECK:      Edges in function: add
;
; CHECK:      Edges in function: sub
;
; CHECK:      RefSCC with 1 call SCCs:
; CHECK-NEXT:   SCC with 1 functions:
; CHECK-NEXT:     sub
;
; CHECK:      RefSCC with 1 call SCCs:
; CHECK-NEXT:   SCC with 1 functions:
; CHECK-NEXT:     add
;
; CHECK:      RefSCC with 1 call SCCs:
; CHECK-NEXT:   SCC with 1 functions:
; CHECK-NEXT:     main

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
