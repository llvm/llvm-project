; RUN: llc -mtriple=riscv32 -mattr=+d -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32-LP64
; RUN: llc -mtriple=riscv64 -mattr=+d -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32-LP64
; RUN: llc -mtriple=riscv32 -mattr=+d -target-abi ilp32d -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32D-LP64D
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64d -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=ILP32D-LP64D

@var = global [32 x double] zeroinitializer

; All floating point registers are temporaries for the ilp32 and lp64 ABIs.
; fs0-fs11 are callee-saved for the ilp32f, ilp32d, lp64f, and lp64d ABIs.

; This function tests that RISCVRegisterInfo::getCalleeSavedRegs returns
; something appropriate.

define void @callee() nounwind {
; ILP32-LP64-LABEL: callee:
; ILP32-LP64:       # %bb.0:
; ILP32-LP64-NEXT:    lui a0, %hi(var)
; ILP32-LP64-NEXT:    fld ft0, %lo(var)(a0)
; ILP32-LP64-NEXT:    addi a1, a0, %lo(var)
; ILP32-LP64-NEXT:    fld ft1, 8(a1)
; ILP32-LP64-NEXT:    fld ft2, 16(a1)
; ILP32-LP64-NEXT:    fld ft3, 24(a1)
; ILP32-LP64-NEXT:    fld ft4, 32(a1)
; ILP32-LP64-NEXT:    fld ft5, 40(a1)
; ILP32-LP64-NEXT:    fld ft6, 48(a1)
; ILP32-LP64-NEXT:    fld ft7, 56(a1)
; ILP32-LP64-NEXT:    fld fa0, 64(a1)
; ILP32-LP64-NEXT:    fld fa1, 72(a1)
; ILP32-LP64-NEXT:    fld fa2, 80(a1)
; ILP32-LP64-NEXT:    fld fa3, 88(a1)
; ILP32-LP64-NEXT:    fld fa4, 96(a1)
; ILP32-LP64-NEXT:    fld fa5, 104(a1)
; ILP32-LP64-NEXT:    fld fa6, 112(a1)
; ILP32-LP64-NEXT:    fld fa7, 120(a1)
; ILP32-LP64-NEXT:    fld ft8, 128(a1)
; ILP32-LP64-NEXT:    fld ft9, 136(a1)
; ILP32-LP64-NEXT:    fld ft10, 144(a1)
; ILP32-LP64-NEXT:    fld ft11, 152(a1)
; ILP32-LP64-NEXT:    fld fs0, 160(a1)
; ILP32-LP64-NEXT:    fld fs1, 168(a1)
; ILP32-LP64-NEXT:    fld fs2, 176(a1)
; ILP32-LP64-NEXT:    fld fs3, 184(a1)
; ILP32-LP64-NEXT:    fld fs4, 192(a1)
; ILP32-LP64-NEXT:    fld fs5, 200(a1)
; ILP32-LP64-NEXT:    fld fs6, 208(a1)
; ILP32-LP64-NEXT:    fld fs7, 216(a1)
; ILP32-LP64-NEXT:    fld fs8, 248(a1)
; ILP32-LP64-NEXT:    fld fs9, 240(a1)
; ILP32-LP64-NEXT:    fld fs10, 232(a1)
; ILP32-LP64-NEXT:    fld fs11, 224(a1)
; ILP32-LP64-NEXT:    fsd fs8, 248(a1)
; ILP32-LP64-NEXT:    fsd fs9, 240(a1)
; ILP32-LP64-NEXT:    fsd fs10, 232(a1)
; ILP32-LP64-NEXT:    fsd fs11, 224(a1)
; ILP32-LP64-NEXT:    fsd fs7, 216(a1)
; ILP32-LP64-NEXT:    fsd fs6, 208(a1)
; ILP32-LP64-NEXT:    fsd fs5, 200(a1)
; ILP32-LP64-NEXT:    fsd fs4, 192(a1)
; ILP32-LP64-NEXT:    fsd fs3, 184(a1)
; ILP32-LP64-NEXT:    fsd fs2, 176(a1)
; ILP32-LP64-NEXT:    fsd fs1, 168(a1)
; ILP32-LP64-NEXT:    fsd fs0, 160(a1)
; ILP32-LP64-NEXT:    fsd ft11, 152(a1)
; ILP32-LP64-NEXT:    fsd ft10, 144(a1)
; ILP32-LP64-NEXT:    fsd ft9, 136(a1)
; ILP32-LP64-NEXT:    fsd ft8, 128(a1)
; ILP32-LP64-NEXT:    fsd fa7, 120(a1)
; ILP32-LP64-NEXT:    fsd fa6, 112(a1)
; ILP32-LP64-NEXT:    fsd fa5, 104(a1)
; ILP32-LP64-NEXT:    fsd fa4, 96(a1)
; ILP32-LP64-NEXT:    fsd fa3, 88(a1)
; ILP32-LP64-NEXT:    fsd fa2, 80(a1)
; ILP32-LP64-NEXT:    fsd fa1, 72(a1)
; ILP32-LP64-NEXT:    fsd fa0, 64(a1)
; ILP32-LP64-NEXT:    fsd ft7, 56(a1)
; ILP32-LP64-NEXT:    fsd ft6, 48(a1)
; ILP32-LP64-NEXT:    fsd ft5, 40(a1)
; ILP32-LP64-NEXT:    fsd ft4, 32(a1)
; ILP32-LP64-NEXT:    fsd ft3, 24(a1)
; ILP32-LP64-NEXT:    fsd ft2, 16(a1)
; ILP32-LP64-NEXT:    fsd ft1, 8(a1)
; ILP32-LP64-NEXT:    fsd ft0, %lo(var)(a0)
; ILP32-LP64-NEXT:    ret
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
; ILP32D-LP64D-NEXT:    fld ft0, %lo(var)(a0)
; ILP32D-LP64D-NEXT:    addi a1, a0, %lo(var)
  %val = load [32 x double], [32 x double]* @var
  store volatile [32 x double] %val, [32 x double]* @var
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
; ILP32F-LP64D-LABEL: caller:
; ILP32D-LP64D: fld	fs8, 160(s1)
; ILP32D-LP64D-NEXT: fld fs9, 168(s1)
; ILP32D-LP64D-NEXT: fld fs10, 176(s1)
; ILP32D-LP64D-NEXT: fld fs11, 184(s1)
; ILP32D-LP64D-NEXT: fld fs0, 192(s1)
; ILP32D-LP64D-NEXT: fld fs1, 200(s1)
; ILP32D-LP64D-NEXT: fld fs2, 208(s1)
; ILP32D-LP64D-NEXT: fld fs3, 216(s1)
; ILP32D-LP64D-NEXT: fld fs4, 224(s1)
; ILP32D-LP64D-NEXT: fld fs5, 232(s1)
; ILP32D-LP64D-NEXT: fld fs6, 240(s1)
; ILP32D-LP64D-NEXT: fld fs7, 248(s1)
; ILP32D-LP64D-NEXT: call	callee
; ILP32D-LP64D-NEXT: fsd fs7, 248(s1)
; ILP32D-LP64D-NEXT: fsd fs6, 240(s1)
; ILP32D-LP64D-NEXT: fsd fs5, 232(s1)
; ILP32D-LP64D-NEXT: fsd fs4, 224(s1)
; ILP32D-LP64D-NEXT: fsd fs3, 216(s1)
; ILP32D-LP64D-NEXT: fsd fs2, 208(s1)
; ILP32D-LP64D-NEXT: fsd fs1, 200(s1)
; ILP32D-LP64D-NEXT: fsd fs0, 192(s1)
; ILP32D-LP64D-NEXT: fsd fs11, 184(s1)
; ILP32D-LP64D-NEXT: fsd fs10, 176(s1)
; ILP32D-LP64D-NEXT: fsd fs9, 168(s1)
; ILP32D-LP64D-NEXT: fsd fs8, 160(s1)
; ILP32D-LP64D-NEXT: fld ft0, {{[0-9]+}}(sp)
  %val = load [32 x double], [32 x double]* @var
  call void @callee()
  store volatile [32 x double] %val, [32 x double]* @var
  ret void
}
