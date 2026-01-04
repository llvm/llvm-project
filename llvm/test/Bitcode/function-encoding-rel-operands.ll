; Basic correctness test to check that instruction operands are encoded with
; relative IDs.
; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s
; RUN: verify-uselistorder < %s

; CHECK: FUNCTION_BLOCK
; CHECK: INST_UNOP {{.*}}op0=1
; CHECK: INST_RET {{.*}}op0=1
define double @test_float_unops(double %a) nounwind {
  %1 = fneg double %a
  ret double %1
}


; CHECK: FUNCTION_BLOCK
; CHECK: INST_BINOP {{.*}}op0=1 op1=1
; CHECK: INST_BINOP {{.*}}op0=1 op1=1
; CHECK: INST_BINOP {{.*}}op0=1 op1=1
; CHECK: INST_RET {{.*}}op0=1
define i32 @test_int_binops(i32 %a) nounwind {
entry:
  %0 = add i32 %a, %a
  %1 = sub i32 %0, %0
  %2 = mul i32 %1, %1
  ret i32 %2
}


; CHECK: FUNCTION_BLOCK
; CHECK: INST_CAST {{.*}}op0=1
; CHECK: INST_BINOP {{.*}}op0=1 op1=1
; CHECK: INST_BINOP {{.*}}op0=1 op1=1
; CHECK: INST_BINOP {{.*}}op0=1 op1=1
; CHECK: INST_BINOP {{.*}}op0=1 op1=1
; CHECK: INST_RET {{.*}}op0=1
define double @test_float_binops(i32 %a) nounwind {
  %1 = sitofp i32 %a to double
  %2 = fadd double %1, %1
  %3 = fsub double %2, %2
  %4 = fmul double %3, %3
  %5 = fdiv double %4, %4
  ret double %5
}


; CHECK: FUNCTION_BLOCK
; skip checking operands of INST_GEP since that depends on ordering
; between literals and the formal parameters.
; CHECK: INST_GEP {{.*}}
; CHECK: INST_LOAD {{.*}}op0=1 {{.*}}
; CHECK: INST_CMP2 {{.*}}op0=1 {{.*}}
; CHECK: INST_RET {{.*}}op0=1
define i1 @test_load(i32 %a, {i32, i32}* %ptr) nounwind {
entry:
  %0 = getelementptr inbounds {i32, i32}, {i32, i32}* %ptr, i32 %a, i32 0
  %1 = load i32, i32* %0
  %2 = icmp eq i32 %1, %a
  ret i1 %2
}

; CHECK: Stream type: LLVM IR
