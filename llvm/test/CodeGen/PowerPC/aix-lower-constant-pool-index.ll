; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=small -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=32SMALL-MIR %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=large -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=32LARGE-MIR %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -code-model=small -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=64SMALL-MIR %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -code-model=large -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=64LARGE-MIR %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=small < %s | FileCheck --check-prefixes=32SMALL-ASM,CHECK %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=large < %s | FileCheck --check-prefixes=32LARGE-ASM,CHECK %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -code-model=small < %s | FileCheck --check-prefixes=64SMALL-ASM,CHECK %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -code-model=large < %s | FileCheck --check-prefixes=64LARGE-ASM,CHECK %s

define float @test_float() {
entry:
  ret float 5.500000e+00
}

; 32SMALL-MIR: renamable $r[[REG1:[0-9]+]] = LWZtoc %const.0, $r2 :: (load 4 from got)
; 32SMALL-MIR: renamable $f[[REG2:[0-9]+]] = LFS 0, killed renamable $r[[REG1]] :: (load 4 from constant-pool)

; 32LARGE-MIR: renamable $r[[REG1:[0-9]+]] = ADDIStocHA $r2, %const.0
; 32LARGE-MIR: renamable $r[[REG2:[0-9]+]] = LWZtocL %const.0, killed renamable $r[[REG1]], implicit $r2 :: (load 4 from got)
; 32LARGE-MIR: renamable $f[[REG3:[0-9]+]] = LFS 0, killed renamable $r[[REG2]] :: (load 4 from constant-pool)

; 64SMALL-MIR: renamable $x[[REG1:[0-9]+]] = LDtocCPT %const.0, $x2 :: (load 8 from got)
; 64SMALL-MIR: renamable $f[[REG2:[0-9]+]] = LFS 0, killed renamable $x[[REG1]] :: (load 4 from constant-pool)

; 64LARGE-MIR: renamable $x[[REG1:[0-9]+]] = ADDIStocHA8 $x2, %const.0
; 64LARGE-MIR: renamable $x[[REG2:[0-9]+]] = LDtocL %const.0, killed renamable $x[[REG1]], implicit $x2 :: (load 8 from got)
; 64LARGE-MIR: renamable $f[[REG3:[0-9]+]] = LFS 0, killed renamable $x[[REG2]] :: (load 4 from constant-pool)

; 32SMALL-ASM:         .csect .rodata[RO],2
; 32SMALL-ASM:         .align  2
; 32SMALL-ASM: L..CPI0_0:
; 32SMALL-ASM:         .vbyte	4, 0x40b00000
; 32SMALL-ASM: .test_float:
; 32SMALL-ASM:         lwz [[REG1:[0-9]+]], L..C0(2)
; 32SMALL-ASM:         lfs 1, 0([[REG1]])
; 32SMALL-ASM:         blr

; 32LARGE-ASM:         .csect .rodata[RO],2
; 32LARGE-ASM:         .align  2
; 32LARGE-ASM: L..CPI0_0:
; 32LARGE-ASM:         .vbyte	4, 0x40b00000
; 32LARGE-ASM: .test_float:
; 32LARGE-ASM:         addis [[REG1:[0-9]+]], L..C0@u(2)
; 32LARGE-ASM:         lwz [[REG2:[0-9]+]], L..C0@l([[REG1]])
; 32LARGE-ASM:         lfs 1, 0([[REG2]])
; 32LARGE-ASM:         blr

; 64SMALL-ASM:         .csect .rodata[RO],2
; 64SMALL-ASM:         .align  2
; 64SMALL-ASM: L..CPI0_0:
; 64SMALL-ASM:         .vbyte	4, 0x40b00000
; 64SMALL-ASM: .test_float:
; 64SMALL-ASM:         ld [[REG1:[0-9]+]], L..C0(2)
; 64SMALL-ASM:         lfs 1, 0([[REG1]])
; 64SMALL-ASM:         blr

; 64LARGE-ASM:         .csect .rodata[RO],2
; 64LARGE-ASM:         .align  2
; 64LARGE-ASM: L..CPI0_0:
; 64LARGE-ASM:         .vbyte	4, 0x40b00000
; 64LARGE-ASM: .test_float:
; 64LARGE-ASM:         addis [[REG1:[0-9]+]], L..C0@u(2)
; 64LARGE-ASM:         ld [[REG2:[0-9]+]], L..C0@l([[REG1]])
; 64LARGE-ASM:         lfs 1, 0([[REG2]])
; 64LARGE-ASM:         blr

; CHECK: .toc
; CHECK: .tc L..CPI0_0[TC],L..CPI0_0
