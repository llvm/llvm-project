; RUN: llc -mtriple=riscv32 -mattr=+f -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32-LP64
; RUN: llc -mtriple=riscv64 -mattr=+f -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32-LP64

@var = global [32 x float] zeroinitializer

; All floating point registers are temporaries for the ilp32 and lp64 ABIs.

; This function tests that RISCVRegisterInfo::getCalleeSavedRegs returns
; something appropriate.

define void @callee() {
; ILP32-LP64-LABEL: callee:
; ILP32-LP64:       # %bb.0:
; ILP32-LP64-NEXT:    lui a0, %hi(var)
; ILP32-LP64-NEXT:    addi a1, a0, %lo(var)
; ILP32-LP64-NEXT:    flw ft0, %lo(var)(a0)
; ILP32-LP64-NEXT:    flw ft1, 4(a1)
; ILP32-LP64-NEXT:    flw ft2, 8(a1)
; ILP32-LP64-NEXT:    flw ft3, 12(a1)
; ILP32-LP64-NEXT:    flw ft4, 16(a1)
; ILP32-LP64-NEXT:    flw ft5, 20(a1)
; ILP32-LP64-NEXT:    flw ft6, 24(a1)
; ILP32-LP64-NEXT:    flw ft7, 28(a1)
; ILP32-LP64-NEXT:    flw fa0, 32(a1)
; ILP32-LP64-NEXT:    flw fa1, 36(a1)
; ILP32-LP64-NEXT:    flw fa2, 40(a1)
; ILP32-LP64-NEXT:    flw fa3, 44(a1)
; ILP32-LP64-NEXT:    flw fa4, 48(a1)
; ILP32-LP64-NEXT:    flw fa5, 52(a1)
; ILP32-LP64-NEXT:    flw fa6, 56(a1)
; ILP32-LP64-NEXT:    flw fa7, 60(a1)
; ILP32-LP64-NEXT:    flw ft8, 64(a1)
; ILP32-LP64-NEXT:    flw ft9, 68(a1)
; ILP32-LP64-NEXT:    flw ft10, 72(a1)
; ILP32-LP64-NEXT:    flw ft11, 76(a1)
; ILP32-LP64-NEXT:    flw fs0, 80(a1)
; ILP32-LP64-NEXT:    flw fs1, 84(a1)
; ILP32-LP64-NEXT:    flw fs2, 88(a1)
; ILP32-LP64-NEXT:    flw fs3, 92(a1)
; ILP32-LP64-NEXT:    flw fs4, 96(a1)
; ILP32-LP64-NEXT:    flw fs5, 100(a1)
; ILP32-LP64-NEXT:    flw fs6, 104(a1)
; ILP32-LP64-NEXT:    flw fs7, 108(a1)
; ILP32-LP64-NEXT:    flw fs8, 112(a1)
; ILP32-LP64-NEXT:    flw fs9, 116(a1)
; ILP32-LP64-NEXT:    flw fs10, 120(a1)
; ILP32-LP64-NEXT:    flw fs11, 124(a1)
; ILP32-LP64-NEXT:    fsw fs11, 124(a1)
; ILP32-LP64-NEXT:    fsw fs10, 120(a1)
; ILP32-LP64-NEXT:    fsw fs9, 116(a1)
; ILP32-LP64-NEXT:    fsw fs8, 112(a1)
; ILP32-LP64-NEXT:    fsw fs7, 108(a1)
; ILP32-LP64-NEXT:    fsw fs6, 104(a1)
; ILP32-LP64-NEXT:    fsw fs5, 100(a1)
; ILP32-LP64-NEXT:    fsw fs4, 96(a1)
; ILP32-LP64-NEXT:    fsw fs3, 92(a1)
; ILP32-LP64-NEXT:    fsw fs2, 88(a1)
; ILP32-LP64-NEXT:    fsw fs1, 84(a1)
; ILP32-LP64-NEXT:    fsw fs0, 80(a1)
; ILP32-LP64-NEXT:    fsw ft11, 76(a1)
; ILP32-LP64-NEXT:    fsw ft10, 72(a1)
; ILP32-LP64-NEXT:    fsw ft9, 68(a1)
; ILP32-LP64-NEXT:    fsw ft8, 64(a1)
; ILP32-LP64-NEXT:    fsw fa7, 60(a1)
; ILP32-LP64-NEXT:    fsw fa6, 56(a1)
; ILP32-LP64-NEXT:    fsw fa5, 52(a1)
; ILP32-LP64-NEXT:    fsw fa4, 48(a1)
; ILP32-LP64-NEXT:    fsw fa3, 44(a1)
; ILP32-LP64-NEXT:    fsw fa2, 40(a1)
; ILP32-LP64-NEXT:    fsw fa1, 36(a1)
; ILP32-LP64-NEXT:    fsw fa0, 32(a1)
; ILP32-LP64-NEXT:    fsw ft7, 28(a1)
; ILP32-LP64-NEXT:    fsw ft6, 24(a1)
; ILP32-LP64-NEXT:    fsw ft5, 20(a1)
; ILP32-LP64-NEXT:    fsw ft4, 16(a1)
; ILP32-LP64-NEXT:    fsw ft3, 12(a1)
; ILP32-LP64-NEXT:    fsw ft2, 8(a1)
; ILP32-LP64-NEXT:    fsw ft1, 4(a1)
; ILP32-LP64-NEXT:    fsw ft0, %lo(var)(a0)
; ILP32-LP64-NEXT:    ret
  %val = load [32 x float], [32 x float]* @var
  store volatile [32 x float] %val, [32 x float]* @var
  ret void
}

; This function tests that RISCVRegisterInfo::getCallPreservedMask returns
; something appropriate.
;
; For the soft float ABIs, no floating point registers are preserved, and
; codegen will use only ft0 in the body of caller.

define void @caller() {
; ILP32-LP64-LABEL: caller:
; ILP32-LP64-NOT:     ft{{[1-9][0-9]*}}
; ILP32-LP64-NOT:     fs{{[0-9]+}}
; ILP32-LP64-NOT:     fa{{[0-9]+}}
; ILP32-LP64:         ret
; ILP32-LP64-NOT:     ft{{[1-9][0-9]*}}
; ILP32-LP64-NOT:     fs{{[0-9]+}}
; ILP32-LP64-NOT:     fa{{[0-9]+}}
  %val = load [32 x float], [32 x float]* @var
  call void @callee()
  store volatile [32 x float] %val, [32 x float]* @var
  ret void
}
