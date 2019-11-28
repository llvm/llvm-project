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

define void @foo() {
entry:
  %tmp = alloca i64
  br label %__here

__here:
  store i64 ptrtoint (i8* blockaddress(@foo, %__here) to i64), i64* %tmp
  ret void
}

; 32SMALL-MIR: renamable $r[[REG1:[0-9]+]] = LWZtoc blockaddress(@foo, %ir-block.__here), $r2 :: (load 4 from got)

; 32LARGE-MIR: renamable $r[[REG1:[0-9]+]] = ADDIStocHA $r2, blockaddress(@foo, %ir-block.__here)
; 32LARGE-MIR: renamable $r[[REG2:[0-9]+]] = LWZtocL blockaddress(@foo, %ir-block.__here), killed renamable $r[[REG1]], implicit $r2 :: (load 4 from got)

; 64SMALL-MIR: renamable $x[[REG1:[0-9]+]] = LDtocBA blockaddress(@foo, %ir-block.__here), $x2 :: (load 8 from got)

; 64LARGE-MIR: renamable $x[[REG1:[0-9]+]] = ADDIStocHA8 $x2, blockaddress(@foo, %ir-block.__here)
; 64LARGE-MIR: renamable $x[[REG2:[0-9]+]] = LDtocL blockaddress(@foo, %ir-block.__here), killed renamable $x[[REG1]], implicit $x2 :: (load 8 from got)

; 32SMALL-ASM-LABEL: foo
; 32SMALL-ASM: .foo:
; 32SMALL-ASM: Ltmp0:
; 32SMALL-ASM: 	       lwz [[REG1:[0-9]+]], LC0(2)

; 32LARGE-ASM-LABEL: foo
; 32LARGE-ASM: .foo:
; 32LARGE-ASM: Ltmp0:
; 32LARGE-ASM:         addis [[REG1:[0-9]+]], LC0@u(2)
; 32LARGE-ASM:         lwz [[REG2:[0-9]+]], LC0@l([[REG1]])

; 64SMALL-ASM-LABEL: foo
; 64SMALL-ASM: .foo:
; 64SMALL-ASM: Ltmp0:
; 64SMALL-ASM:         ld [[REG1:[0-9]+]], LC0(2)

; 64LARGE-ASM-LABEL: foo
; 64LARGE-ASM: .foo:
; 64LARGE-ASM: Ltmp0:
; 64LARGE-ASM:         addis [[REG1:[0-9]+]], LC0@u(2)
; 64LARGE-ASM:         ld [[REG2:[0-9]+]], LC0@l([[REG1]])

; CHECK: .toc
; CHECK: .tc Ltmp0[TC],Ltmp0
