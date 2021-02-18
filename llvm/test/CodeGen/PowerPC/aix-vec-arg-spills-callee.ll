; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec \
; RUN:     -vec-extabi -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=32BIT

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec -vec-extabi \
; RUN:     -stop-after=machine-cp -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=MIR32

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec \
; RUN:     -vec-extabi -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=64BIT

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec -vec-extabi \
; RUN:     -stop-after=machine-cp -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=MIR64

%struct.Test = type { double, double, double, double }

define double @test(i32 signext %r3, i32 signext %r4, double %fpr1, double %fpr2, <2 x double> %v2, <2 x double> %v3, <2 x double> %v4, <2 x double> %v5, <2 x double> %v6, <2 x double> %v7, <2 x double> %v8, <2 x double> %v9, <2 x double> %v10, <2 x double> %v11, <2 x double> %v12, <2 x double> %v13, <2 x double> %vSpill, double %fpr3, double %fpr4, double %fpr5, double %fpr6, double %fpr7, double %fpr8, double %fpr9, double %fpr10, double %fpr11, double %fpr12, double %fpr13, i32 signext %gprSpill, %struct.Test* nocapture readonly byval(%struct.Test) align 4 %t) {
entry:
  %vecext = extractelement <2 x double> %vSpill, i32 0
  %x = getelementptr inbounds %struct.Test, %struct.Test* %t, i32 0, i32 0
  %0 = load double, double* %x, align 4
  %add = fadd double %vecext, %0
  ret double %add
}

; 32BIT-LABEL: .test:
; 32BIT-DAG:     lfd {{[0-9]+}}, 48(1)
; 32BIT-DAG:     lfd {{[0-9]+}}, 156(1)

; MIR32: name:            test
; MIR32: fixedStack:
; MIR32:   - { id: 0, type: default, offset: 156, size: 32, alignment: 4, stack-id: default,
; MIR32:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; MIR32:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR32:   - { id: 1, type: default, offset: 152, size: 4, alignment: 8, stack-id: default,
; MIR32:       isImmutable: true, isAliased: false, callee-saved-register: '', callee-saved-restored: true,
; MIR32:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR32:   - { id: 2, type: default, offset: 48, size: 16, alignment: 16, stack-id: default,
; MIR32:       isImmutable: true, isAliased: false, callee-saved-register: '', callee-saved-restored: true,
; MIR32:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }

; MIR32:  renamable $[[GPR1:r[0-9]+]] = ADDI %fixed-stack.2, 0
; MIR32:  renamable $[[GPR2:r[0-9]+]] = ADDI %fixed-stack.0, 0
; MIR32:  renamable $f{{[0-9]+}} = XFLOADf64 $zero, killed renamable $[[GPR1]]
; MIR32:  renamable $f{{[0-9]+}} = XFLOADf64 $zero, killed renamable $[[GPR2]]

; 64BIT-LABEL: .test:
; 64BIT-DAG:     lfd {{[0-9]+}}, 80(1)
; 64BIT-DAG:     lfd {{[0-9]+}}, 192(1)

; MIR64: name:            test
; MIR64: fixedStack:
; MIR64:   - { id: 0, type: default, offset: 192, size: 32, alignment: 16, stack-id: default,
; MIR64:       isImmutable: false, isAliased: true, callee-saved-register: '', callee-saved-restored: true,
; MIR64:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR64:   - { id: 1, type: default, offset: 188, size: 4, alignment: 4, stack-id: default,
; MIR64:       isImmutable: true, isAliased: false, callee-saved-register: '', callee-saved-restored: true,
; MIR64:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; MIR64:   - { id: 2, type: default, offset: 80, size: 16, alignment: 16, stack-id: default,
; MIR64:       isImmutable: true, isAliased: false, callee-saved-register: '', callee-saved-restored: true,
; MIR64:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }

; MIR64:   renamable $[[GPR1:x[0-9]+]] = ADDI8 %fixed-stack.2, 0
; MIR64:   renamable $[[GPR2:x[0-9]+]] = ADDI8 %fixed-stack.0, 0
; MIR64:   renamable $f{{[0-9]+}} = XFLOADf64 $zero8, killed renamable $[[GPR1]]
; MIR64:   renamable $f{{[0-9]+}} = XFLOADf64 $zero8, killed renamable $[[GPR2]]
