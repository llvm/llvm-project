; RUN: llc -mtriple=riscv32 -mattr=+f -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32-LP64
; RUN: llc -mtriple=riscv64 -mattr=+f -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32-LP64
; RUN: llc -mtriple=riscv32 -mattr=+f -target-abi ilp32f -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32F-LP64F
; RUN: llc -mtriple=riscv64 -mattr=+f -target-abi lp64f -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32F-LP64F
; RUN: llc -mtriple=riscv32 -mattr=+d -target-abi ilp32d -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32D-LP64D
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64d -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32D-LP64D

@var = global [32 x float] zeroinitializer

; All floating point registers are temporaries for the ilp32 and lp64 ABIs.
; fs0-fs11 are callee-saved for the ilp32f, ilp32d, lp64f, and lp64d ABIs.

; This function tests that RISCVRegisterInfo::getCalleeSavedRegs returns
; something appropriate.

define void @callee() nounwind {
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
;
; ILP32F-LP64F-LABEL: callee:
; ILP32F-LP64F:       # %bb.0:
; ILP32F-LP64F-NEXT:    addi sp, sp, -48
; ILP32F-LP64F-NEXT:    fsw fs0, 44(sp)
; ILP32F-LP64F-NEXT:    fsw fs1, 40(sp)
; ILP32F-LP64F-NEXT:    fsw fs2, 36(sp)
; ILP32F-LP64F-NEXT:    fsw fs3, 32(sp)
; ILP32F-LP64F-NEXT:    fsw fs4, 28(sp)
; ILP32F-LP64F-NEXT:    fsw fs5, 24(sp)
; ILP32F-LP64F-NEXT:    fsw fs6, 20(sp)
; ILP32F-LP64F-NEXT:    fsw fs7, 16(sp)
; ILP32F-LP64F-NEXT:    fsw fs8, 12(sp)
; ILP32F-LP64F-NEXT:    fsw fs9, 8(sp)
; ILP32F-LP64F-NEXT:    fsw fs10, 4(sp)
; ILP32F-LP64F-NEXT:    fsw fs11, 0(sp)
; ILP32F-LP64F-NEXT:    lui a0, %hi(var)
; ILP32F-LP64F-NEXT:    addi a1, a0, %lo(var)
;
; ILP32D-LP64D-LABEL: callee:
; ILP32D-LP64D:       # %bb.0:
; ILP32D-LP64D-NEXT:    addi sp, sp, -96
; ILP32D-LP64D-NEXT:    fsd fs0, 88(sp)
; ILP32D-LP64D-NEXT:    fsd fs1, 80(sp)
; ILP32D-LP64D-NEXT:    fsd fs2, 72(sp)
; ILP32D-LP64D-NEXT:    fsd fs3, 64(sp)
; ILP32D-LP64D-NEXT:    fsd fs4, 56(sp)
; ILP32D-LP64D-NEXT:    fsd fs5, 48(sp)
; ILP32D-LP64D-NEXT:    fsd fs6, 40(sp)
; ILP32D-LP64D-NEXT:    fsd fs7, 32(sp)
; ILP32D-LP64D-NEXT:    fsd fs8, 24(sp)
; ILP32D-LP64D-NEXT:    fsd fs9, 16(sp)
; ILP32D-LP64D-NEXT:    fsd fs10, 8(sp)
; ILP32D-LP64D-NEXT:    fsd fs11, 0(sp)
; ILP32D-LP64D-NEXT:    lui a0, %hi(var)
; ILP32D-LP64D-NEXT:    addi a1, a0, %lo(var)
  %val = load [32 x float], [32 x float]* @var
  store volatile [32 x float] %val, [32 x float]* @var
  ret void
}

; This function tests that RISCVRegisterInfo::getCallPreservedMask returns
; something appropriate.
;
; For the soft float ABIs, no floating point registers are preserved, and
; codegen will use only ft0 in the body of caller. For the 'f' and 'd ABIs,
; fs0-fs11 are preserved across calls.

define void @caller() nounwind {
; ILP32-LP64-LABEL: caller:
; ILP32-LP64-NOT:     ft{{[1-9][0-9]*}}
; ILP32-LP64-NOT:     fs{{[0-9]+}}
; ILP32-LP64-NOT:     fa{{[0-9]+}}
; ILP32-LP64:         call callee
; ILP32-LP64-NOT:     ft{{[1-9][0-9]*}}
; ILP32-LP64-NOT:     fs{{[0-9]+}}
; ILP32-LP64-NOT:     fa{{[0-9]+}}
; ILP32-LP64:         ret
;
; ILP32F-LP64F-LABEL: caller:
; ILP32F-LP64F:       flw fs8, 80(s1)
; ILP32F-LP64F-NEXT:  flw fs9, 84(s1)
; ILP32F-LP64F-NEXT:  flw fs10, 88(s1)
; ILP32F-LP64F-NEXT:  flw fs11, 92(s1)
; ILP32F-LP64F-NEXT:  flw fs0, 96(s1)
; ILP32F-LP64F-NEXT:  flw fs1, 100(s1)
; ILP32F-LP64F-NEXT:  flw fs2, 104(s1)
; ILP32F-LP64F-NEXT:  flw fs3, 108(s1)
; ILP32F-LP64F-NEXT:  flw fs4, 112(s1)
; ILP32F-LP64F-NEXT:  flw fs5, 116(s1)
; ILP32F-LP64F-NEXT:  flw fs6, 120(s1)
; ILP32F-LP64F-NEXT:  flw fs7, 124(s1)
; ILP32F-LP64F-NEXT:  call callee
; ILP32F-LP64F-NEXT:  fsw fs7, 124(s1)
; ILP32F-LP64F-NEXT:  fsw fs6, 120(s1)
; ILP32F-LP64F-NEXT:  fsw fs5, 116(s1)
; ILP32F-LP64F-NEXT:  fsw fs4, 112(s1)
; ILP32F-LP64F-NEXT:  fsw fs3, 108(s1)
; ILP32F-LP64F-NEXT:  fsw fs2, 104(s1)
; ILP32F-LP64F-NEXT:  fsw fs1, 100(s1)
; ILP32F-LP64F-NEXT:  fsw fs0, 96(s1)
; ILP32F-LP64F-NEXT:  fsw fs11, 92(s1)
; ILP32F-LP64F-NEXT:  fsw fs10, 88(s1)
; ILP32F-LP64F-NEXT:  fsw fs9, 84(s1)
; ILP32F-LP64F-NEXT:  fsw fs8, 80(s1)
; ILP32F-LP64F-NEXT:  lw ft0, {{[0-9]+}}(sp)
;
; ILP32D-LP64D-LABEL: caller:
; ILP32D-LP64D:       flw fs8, 80(s1)
; ILP32D-LP64D-NEXT:  flw fs9, 84(s1)
; ILP32D-LP64D-NEXT:  flw fs10, 88(s1)
; ILP32D-LP64D-NEXT:  flw fs11, 92(s1)
; ILP32D-LP64D-NEXT:  flw fs0, 96(s1)
; ILP32D-LP64D-NEXT:  flw fs1, 100(s1)
; ILP32D-LP64D-NEXT:  flw fs2, 104(s1)
; ILP32D-LP64D-NEXT:  flw fs3, 108(s1)
; ILP32D-LP64D-NEXT:  flw fs4, 112(s1)
; ILP32D-LP64D-NEXT:  flw fs5, 116(s1)
; ILP32D-LP64D-NEXT:  flw fs6, 120(s1)
; ILP32D-LP64D-NEXT:  flw fs7, 124(s1)
; ILP32D-LP64D-NEXT:  call callee
; ILP32D-LP64D-NEXT:  fsw fs7, 124(s1)
; ILP32D-LP64D-NEXT:  fsw fs6, 120(s1)
; ILP32D-LP64D-NEXT:  fsw fs5, 116(s1)
; ILP32D-LP64D-NEXT:  fsw fs4, 112(s1)
; ILP32D-LP64D-NEXT:  fsw fs3, 108(s1)
; ILP32D-LP64D-NEXT:  fsw fs2, 104(s1)
; ILP32D-LP64D-NEXT:  fsw fs1, 100(s1)
; ILP32D-LP64D-NEXT:  fsw fs0, 96(s1)
; ILP32D-LP64D-NEXT:  fsw fs11, 92(s1)
; ILP32D-LP64D-NEXT:  fsw fs10, 88(s1)
; ILP32D-LP64D-NEXT:  fsw fs9, 84(s1)
; ILP32D-LP64D-NEXT:  fsw fs8, 80(s1)
; ILP32D-LP64D-NEXT:  flw ft0, {{[0-9]+}}(sp)
  %val = load [32 x float], [32 x float]* @var
  call void @callee()
  store volatile [32 x float] %val, [32 x float]* @var
  ret void
}
