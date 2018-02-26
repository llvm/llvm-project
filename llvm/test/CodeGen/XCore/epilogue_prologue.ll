; RUN: llc < %s -march=xcore | FileCheck %s
; RUN: llc < %s -march=xcore -disable-fp-elim | FileCheck %s -check-prefix=CHECKFP

; When using SP for small frames, we don't need any scratch registers (SR).
; When using SP for large frames, we may need two scratch registers.
; When using FP, for large or small frames, we may need one scratch register.

; FP + small frame: spill FP+SR = entsp 2
; CHECKFP-LABEL: f1:
; CHECKFP: entsp 2
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: retsp 2
;
; !FP + small frame: no spills = no stack adjustment needed
; CHECK-LABEL: f1:
; CHECK: stw lr, sp[0]
; CHECK: ldw lr, sp[0]
; CHECK-NEXT: retsp 0
define void @f1() nounwind {
entry:
  tail call void asm sideeffect "", "~{lr}"() nounwind
  ret void
}


; FP + small frame: spill FP+SR+R0+LR = entsp 3 + extsp 1
; CHECKFP-LABEL: f3:
; CHECKFP: entsp 3
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP-NEXT: stw [[REG:r[4-9]+]], r10[2]
; CHECKFP-NEXT: mov [[REG]], r0
; CHECKFP-NEXT: extsp 1
; CHECKFP-NEXT: bl f2
; CHECKFP-NEXT: ldaw sp, sp[1]
; CHECKFP-NEXT: mov r0, [[REG]]
; CHECKFP-NEXT: ldw [[REG]], r10[2]
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: retsp 3
;
; !FP + small frame: spill R0+LR = entsp 2
; CHECK-LABEL: f3:
; CHECK: entsp 2
; CHECK-NEXT: stw [[REG:r[4-9]+]], sp[1]
; CHECK-NEXT: mov [[REG]], r0
; CHECK-NEXT: bl f2
; CHECK-NEXT: mov r0, [[REG]]
; CHECK-NEXT: ldw [[REG]], sp[1]
; CHECK-NEXT: retsp 2
declare void @f2()
define i32 @f3(i32 %i) nounwind {
entry:
  call void @f2()
  ret i32 %i
}


; FP + large frame: spill FP+SR = entsp 2 + 100000
; CHECKFP-LABEL: f4:
; CHECKFP: entsp 65535
; CHECKFP-NEXT: .cfi_def_cfa_offset 262140
; CHECKFP-NEXT: .cfi_offset 15, 0
; CHECKFP-NEXT: extsp 34467
; CHECKFP-NEXT: .cfi_def_cfa_offset 400008
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: .cfi_offset 10, -400004
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP-NEXT: .cfi_def_cfa_register 10
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: ldaw sp, sp[65535]
; CHECKFP-NEXT: retsp 34467
;
; !FP + large frame: spill SR+SR = entsp 2 + 100000
; CHECK-LABEL: f4:
; CHECK: entsp 65535
; CHECK-NEXT: .cfi_def_cfa_offset 262140
; CHECK-NEXT: .cfi_offset 15, 0
; CHECK-NEXT: extsp 34467
; CHECK-NEXT: .cfi_def_cfa_offset 400008
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: retsp 34467
define void @f4() {
entry:
  %0 = alloca [100000 x i32]
  ret void
}


; FP + large frame: spill FP+SR+R4+LR = entsp 3 + 200000  + extsp 1
; CHECKFP: .section .cp.rodata.cst4,"aMc",@progbits,4
; CHECKFP-NEXT: .p2align 2
; CHECKFP-NEXT: .LCPI[[CNST0:[0-9_]+]]:
; CHECKFP-NEXT: .long 200002
; CHECKFP-NEXT: .LCPI[[CNST1:[0-9_]+]]:
; CHECKFP-NEXT: .long 200001
; CHECKFP-NEXT: .text
; CHECKFP-LABEL: f6:
; CHECKFP: entsp 65535
; CHECKFP-NEXT: .cfi_def_cfa_offset 262140
; CHECKFP-NEXT: .cfi_offset 15, 0
; CHECKFP-NEXT: extsp 65535
; CHECKFP-NEXT: .cfi_def_cfa_offset 524280
; CHECKFP-NEXT: extsp 65535
; CHECKFP-NEXT: .cfi_def_cfa_offset 786420
; CHECKFP-NEXT: extsp 3398
; CHECKFP-NEXT: .cfi_def_cfa_offset 800012
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: .cfi_offset 10, -800008
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP-NEXT: .cfi_def_cfa_register 10
; CHECKFP-NEXT: ldw r1, cp[.LCPI[[CNST0]]]
; CHECKFP-NEXT: stw [[REG:r[4-9]+]], r10[r1]
; CHECKFP-NEXT: .cfi_offset 4, -4
; CHECKFP-NEXT: mov [[REG]], r0
; CHECKFP-NEXT: extsp 1
; CHECKFP-NEXT: ldaw r0, r10[2]
; CHECKFP-NEXT: bl f5
; CHECKFP-NEXT: ldaw sp, sp[1]
; CHECKFP-NEXT: ldw r1, cp[.LCPI3_1]
; CHECKFP-NEXT: ldaw r0, r10[r1]
; CHECKFP-NEXT: extsp 1
; CHECKFP-NEXT: bl f5
; CHECKFP-NEXT: ldaw sp, sp[1]
; CHECKFP-NEXT: mov r0, [[REG]]
; CHECKFP-NEXT: ldw r1, cp[.LCPI[[CNST0]]]
; CHECKFP-NEXT: ldw [[REG]], r10[r1]
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: ldaw sp, sp[65535]
; CHECKFP-NEXT: ldaw sp, sp[65535]
; CHECKFP-NEXT: ldaw sp, sp[65535]
; CHECKFP-NEXT: retsp 3398
;
; !FP + large frame: spill SR+SR+R4+LR = entsp 4 + 200000
; CHECK: .section .cp.rodata.cst4,"aMc",@progbits,4
; CHECK-NEXT: .p2align 2
; CHECK-NEXT: .LCPI[[CNST0:[0-9_]+]]:
; CHECK-NEXT: .long 200003
; CHECK-NEXT: .LCPI[[CNST1:[0-9_]+]]:
; CHECK-NEXT: .long 200002
; CHECK-NEXT: .text
; CHECK-LABEL: f6:
; CHECK: entsp 65535
; CHECK-NEXT: .cfi_def_cfa_offset 262140
; CHECK-NEXT: .cfi_offset 15, 0
; CHECK-NEXT: extsp 65535
; CHECK-NEXT: .cfi_def_cfa_offset 524280
; CHECK-NEXT: extsp 65535
; CHECK-NEXT: .cfi_def_cfa_offset 786420
; CHECK-NEXT: extsp 3399
; CHECK-NEXT: .cfi_def_cfa_offset 800016
; CHECK-NEXT: ldaw r1, sp[0]
; CHECK-NEXT: ldw r2, cp[.LCPI[[CNST0]]]
; CHECK-NEXT: stw [[REG:r[4-9]+]], r1[r2]
; CHECK-NEXT: .cfi_offset 4, -4
; CHECK-NEXT: mov [[REG]], r0
; CHECK-NEXT: ldaw r0, sp[3]
; CHECK-NEXT: bl f5
; CHECK-NEXT: ldaw r0, sp[0]
; CHECK-NEXT: ldw r1, cp[.LCPI[[CNST1]]]
; CHECK-NEXT: ldaw r0, r0[r1]
; CHECK-NEXT: bl f5
; CHECK-NEXT: mov r0, [[REG]]
; CHECK-NEXT: ldaw [[REG]], sp[0]
; CHECK-NEXT: ldw r1, cp[.LCPI[[CNST0]]]
; CHECK-NEXT: ldw [[REG]], [[REG]][r1]
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: retsp 3399
declare void @f5(i32*)
define i32 @f6(i32 %i) {
entry:
  %0 = alloca [200000 x i32]
  %1 = getelementptr inbounds [200000 x i32], [200000 x i32]* %0, i32 0, i32 0
  call void @f5(i32* %1)
  %2 = getelementptr inbounds [200000 x i32], [200000 x i32]* %0, i32 0, i32 199999
  call void @f5(i32* %2)
  ret i32 %i
}

; FP + large frame: spill FP+SR+LR = entsp 2 + 256  + extsp 1
; CHECKFP-LABEL: f8:
; CHECKFP: entsp 258
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP-NEXT: mkmsk [[REG:r[0-9]+]], 8
; CHECKFP-NEXT: ldaw r0, r10{{\[}}[[REG]]{{\]}}
; CHECKFP-NEXT: extsp 1
; CHECKFP-NEXT: bl f5
; CHECKFP-NEXT: ldaw sp, sp[1]
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: retsp 258
;
; !FP + large frame: spill SR+SR+LR = entsp 3 + 256
; CHECK-LABEL: f8:
; CHECK: entsp 257
; CHECK-NEXT: ldaw r0, sp[254]
; CHECK-NEXT: bl f5
; CHECK-NEXT: retsp 257
define void @f8() nounwind {
entry:
  %0 = alloca [256 x i32]
  %1 = getelementptr inbounds [256 x i32], [256 x i32]* %0, i32 0, i32 253
  call void @f5(i32* %1)
  ret void
}

; FP + large frame: spill FP+SR+LR = entsp 2 + 32768  + extsp 1
; CHECKFP-LABEL: f9:
; CHECKFP: entsp 32770
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP-NEXT: ldc [[REG:r[0-9]+]], 32767
; CHECKFP-NEXT: ldaw r0, r10{{\[}}[[REG]]{{\]}}
; CHECKFP-NEXT: extsp 1
; CHECKFP-NEXT: bl f5
; CHECKFP-NEXT: ldaw sp, sp[1]
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: retsp 32770
;
; !FP + large frame: spill SR+SR+LR = entsp 3 + 32768
; CHECK-LABEL: f9:
; CHECK: entsp 32771
; CHECK-NEXT: ldaw r0, sp[32768]
; CHECK-NEXT: bl f5
; CHECK-NEXT: retsp 32771
define void @f9() nounwind {
entry:
  %0 = alloca [32768 x i32]
  %1 = getelementptr inbounds [32768 x i32], [32768 x i32]* %0, i32 0, i32 32765
  call void @f5(i32* %1)
  ret void
}
