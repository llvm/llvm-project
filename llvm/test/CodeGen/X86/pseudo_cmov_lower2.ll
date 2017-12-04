; RUN: llc < %s -mtriple=x86_64-linux-gnu -o - | FileCheck %s 

; This test checks that only a single jae gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.  The tricky part
; of this test is that it tests the special PHI operand rewriting code in
; X86TargetLowering::EmitLoweredSelect.
;
; CHECK-LABEL: foo1:
; CHECK: jae
; CHECK-NOT: jae
define double @foo1(float %p1, double %p2, double %p3) nounwind {
entry:
  %c1 = fcmp oge float %p1, 0.000000e+00
  %d0 = fadd double %p2, 1.25e0
  %d1 = fadd double %p3, 1.25e0
  %d2 = select i1 %c1, double %d0, double %d1
  %d3 = select i1 %c1, double %d2, double %p2
  %d4 = select i1 %c1, double %d3, double %p3
  %d5 = fsub double %d2, %d3
  %d6 = fadd double %d5, %d4
  ret double %d6
}

; This test checks that only a single jae gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.  The tricky part
; of this test is that it tests the special PHI operand rewriting code in
; X86TargetLowering::EmitLoweredSelect.
;
; CHECK-LABEL: foo2:
; CHECK: jae
; CHECK-NOT: jae
define double @foo2(float %p1, double %p2, double %p3) nounwind {
entry:
  %c1 = fcmp oge float %p1, 0.000000e+00
  %d0 = fadd double %p2, 1.25e0
  %d1 = fadd double %p3, 1.25e0
  %d2 = select i1 %c1, double %d0, double %d1
  %d3 = select i1 %c1, double %p2, double %d2
  %d4 = select i1 %c1, double %p3, double %d3
  %d5 = fsub double %d2, %d3
  %d6 = fadd double %d5, %d4
  ret double %d6
}

; This test checks that only a single js gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.  The tricky part
; of this test is that it tests the special PHI operand rewriting code in
; X86TargetLowering::EmitLoweredSelect.  It also tests to make sure all
; the operands of the resulting instructions are from the proper places.
;
; CHECK-LABEL: foo3:
; CHECK:          js
; CHECK-NOT: js
; CHECK-LABEL: # %bb.1:
; CHECK-DAG:      movapd  %xmm2, %xmm1
; CHECK-DAG:      movapd  %xmm2, %xmm0
; CHECK-LABEL:.LBB2_2:
; CHECK:          divsd   %xmm1, %xmm0
; CHECK:          ret
define double @foo3(i32 %p1, double %p2, double %p3,
                             double %p4, double %p5) nounwind {
entry:
  %c1 = icmp slt i32 %p1, 0
  %d2 = select i1 %c1, double %p2, double %p3
  %d3 = select i1 %c1, double %p3, double %p4
  %d4 = select i1 %c1, double %d2, double %d3
  %d5 = fdiv double %d4, %d3
  ret double %d5
}

; This test checks that only a single js gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.  The tricky part
; of this test is that it tests the special PHI operand rewriting code in
; X86TargetLowering::EmitLoweredSelect.  It also tests to make sure all
; the operands of the resulting instructions are from the proper places
; when the "opposite condition" handling code in the compiler is used.
; This should be the same code as foo3 above, because we use the opposite
; condition code in the second two selects, but we also swap the operands
; of the selects to give the same actual computation.
;
; CHECK-LABEL: foo4:
; CHECK:          js
; CHECK-NOT: js
; CHECK-LABEL: # %bb.1:
; CHECK-DAG:      movapd  %xmm2, %xmm1
; CHECK-DAG:      movapd  %xmm2, %xmm0
; CHECK-LABEL:.LBB3_2:
; CHECK:          divsd   %xmm1, %xmm0
; CHECK:          ret
define double @foo4(i32 %p1, double %p2, double %p3,
                             double %p4, double %p5) nounwind {
entry:
  %c1 = icmp slt i32 %p1, 0
  %d2 = select i1 %c1, double %p2, double %p3
  %c2 = icmp sge i32 %p1, 0
  %d3 = select i1 %c2, double %p4, double %p3
  %d4 = select i1 %c2, double %d3, double %d2
  %d5 = fdiv double %d4, %d3
  ret double %d5
}

; This test checks that only a single jae gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.  The tricky part
; of this test is that it tests the special code in CodeGenPrepare.
;
; CHECK-LABEL: foo5:
; CHECK: jae
; CHECK-NOT: jae
define double @foo5(float %p1, double %p2, double %p3) nounwind {
entry:
  %c1 = fcmp oge float %p1, 0.000000e+00
  %d0 = fadd double %p2, 1.25e0
  %d1 = fadd double %p3, 1.25e0
  %d2 = select i1 %c1, double %d0, double %d1, !prof !0
  %d3 = select i1 %c1, double %d2, double %p2, !prof !0
  %d4 = select i1 %c1, double %d3, double %p3, !prof !0
  %d5 = fsub double %d2, %d3
  %d6 = fadd double %d5, %d4
  ret double %d6
}

; We should expand select instructions into 3 conditional branches as their
; condtions are different.
;
; CHECK-LABEL: foo6:
; CHECK: jae
; CHECK: jae
; CHECK: jae
define double @foo6(float %p1, double %p2, double %p3) nounwind {
entry:
  %c1 = fcmp oge float %p1, 0.000000e+00
  %c2 = fcmp oge float %p1, 1.000000e+00
  %c3 = fcmp oge float %p1, 2.000000e+00
  %d0 = fadd double %p2, 1.25e0
  %d1 = fadd double %p3, 1.25e0
  %d2 = select i1 %c1, double %d0, double %d1, !prof !0
  %d3 = select i1 %c2, double %d2, double %p2, !prof !0
  %d4 = select i1 %c3, double %d3, double %p3, !prof !0
  %d5 = fsub double %d2, %d3
  %d6 = fadd double %d5, %d4
  ret double %d6
}

!0 = !{!"branch_weights", i32 1, i32 2000}
