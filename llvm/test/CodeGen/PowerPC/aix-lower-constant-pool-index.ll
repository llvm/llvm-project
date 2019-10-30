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
