; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: df_add:
; CHECK: dfadd
define double @df_add(double %x, double %y) local_unnamed_addr #0 {
entry:
  %add = fadd double %x, %y
  ret double %add
}

; CHECK-LABEL: df_sub:
; CHECK: dfsub
define double @df_sub(double %x, double %y) local_unnamed_addr #0 {
entry:
  %sub = fsub double %x, %y
  ret double %sub
}

attributes #0 = { norecurse nounwind readnone "target-cpu"="hexagonv66" }
