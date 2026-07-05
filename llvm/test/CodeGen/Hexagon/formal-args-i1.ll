; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; This tests validates the fact that the formal arguments of type scalar i1
; (passed using 32-bit register) is converted back to use predicate registers
; CHECK: [[P0:p[0-3]]] = tstbit(r0,#0)
; CHECK: [[R0:r[0-9]+]] = mux([[P0]],#3,r2)
; CHECK: memb(r1+#0) = [[R0]]

target triple = "hexagon"

define void @f0(i1 zeroext %a0, ptr nocapture %a1, i8 %a2) local_unnamed_addr #0 {
entry:
  %v0 = select i1 %a0, i8 3, i8 %a2
  store i8 %v0, ptr %a1, align 1
  ret void
}

attributes #0 = { norecurse nounwind optsize "target-cpu"="hexagonv60" }
