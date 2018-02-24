; RUN: llc -O2 -march=bpfel -mattr=+alu32 < %s | FileCheck %s
;
; long long select_u(unsigned a, unsigned b, long long c, long long d)
; {
;   if (a > b)
;     return c;
;   else
;     return d;
; }
;
; long long select_s(signed a, signed b, long long c, long long d)
; {
;   if (a > b)
;     return c;
;   else
;     return d;
;}
; Function Attrs: norecurse nounwind readnone
define dso_local i64 @select_u(i32 %a, i32 %b, i64 %c, i64 %d) local_unnamed_addr #0 {
; CHECK-LABEL: select_u:
entry:
  %cmp = icmp ugt i32 %a, %b
  %c.d = select i1 %cmp, i64 %c, i64 %d
; CHECK: if r{{[0-9]+}} {{<|>}} r{{[0-9]+}} goto
  ret i64 %c.d
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @select_s(i32 %a, i32 %b, i64 %c, i64 %d) local_unnamed_addr #0 {
; CHECK-LABEL: select_s:
entry:
  %cmp = icmp sgt i32 %a, %b
  %c.d = select i1 %cmp, i64 %c, i64 %d
; CHECK: if r{{[0-9]+}} s{{<|>}} r{{[0-9]+}} goto
  ret i64 %c.d
}
