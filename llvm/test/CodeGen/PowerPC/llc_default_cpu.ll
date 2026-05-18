; Test that the default CPU for the triple powerpc64-unknown-linux-gnu is ppc64.
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -frame-pointer=all -mcpu=ppc | FileCheck %s -check-prefixes=LNX-PPC,LNX-COM
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -frame-pointer=all | FileCheck %s -check-prefixes=LNX-PPC64,LNX-COM
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -frame-pointer=all -mcpu=ppc64 | FileCheck %s -check-prefixes=LNX-PPC64,LNX-COM

; Test that the default CPU for the AIX OS is pwr7.
; RUN: llc < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=ppc | FileCheck %s -check-prefixes=AIX-PPC,AIX-COM
; RUN: llc < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr7 | FileCheck %s -check-prefixes=AIX-PWR7,AIX-COM
; RUN: llc < %s -mtriple=powerpc-ibm-aix-xcoff | FileCheck %s -check-prefixes=AIX-PWR7,AIX-COM

; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff -mcpu=ppc | FileCheck %s -check-prefixes=AIX64-PPC,AIX64-COM-NEXT
; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr7 | FileCheck %s -check-prefixes=AIX64-PWR7,AIX64-COM-NEXT
; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff | FileCheck %s -check-prefixes=AIX64-PWR7,AIX64-COM-NEXT

define i32 @main() {
entry:
  %retval = alloca i32, i32 8191, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0
}

;        LNX-COM: .Lfunc_begin0:
;   LNX-COM-NEXT:      .cfi_startproc
;   LNX-COM-NEXT: # %bb.0:                                # %entry
;   LNX-COM-NEXT:      lis 0, -1
;   LNX-PPC-NEXT:      ori 0, 0, 32704
;   LNX-PPC-NEXT:      std 31, -8(1)
; LNX-PPC64-NEXT:      std 31, -8(1)
; LNX-PPC64-NEXT:      ori 0, 0, 32704
;   LNX-COM-NEXT:      stdux 1, 1, 0
;   LNX-COM-NEXT:      .cfi_def_cfa_offset 32832
;   LNX-COM-NEXT:      .cfi_offset r31, -8
;   LNX-COM-NEXT:      mr      31, 1
;   LNX-COM-NEXT:      .cfi_def_cfa_register r31
;   LNX-COM-NEXT:      li 4, 0
;   LNX-COM-NEXT:      li 3, 0
;   LNX-COM-NEXT:      stw 4, 60(31)
;   LNX-COM-NEXT:      ld 1, 0(1)
;   LNX-COM-NEXT:      ld 31, -8(1)
;   LNX-COM-NEXT:      blr
;   LNX-COM-NEXT:      .long   0
;   LNX-COM-NEXT:      .quad   0
;   LNX-COM-NEXT: .Lfunc_end0:

;       AIX-COM: .main:
;  AIX-COM-NEXT: # %bb.0:                                # %entry
;  AIX-COM-NEXT:      lis 0, -1
;  AIX-COM-NEXT:      ori 0, 0, 32736
;  AIX-COM-NEXT:      stwux 1, 1, 0
;  AIX-PPC-NEXT:      li 4, 0
;  AIX-COM-NEXT:      li 3, 0
;  AIX-PPC-NEXT:      stw 4, 36(1)
; AIX-PWR7-NEXT:      stw 3, 36(1)
;  AIX-COM-NEXT:      lwz 1, 0(1)
;  AIX-COM-NEXT:      blr

;       AIX64-COM: .main:
;  AIX64-COM-NEXT: # %bb.0:                                # %entry
;  AIX64-COM-NEXT:    lis 0, -1
;  AIX64-COM-NEXT:    ori 0, 0, 32720
;  AIX64-COM-NEXT:    stdux 1, 1, 0
;  AIX64-PPC-NEXT:    li 4, 0
;  AIX64-COM-NEXT:    li 3, 0
;  AIX64-PPC-NEXT:    stw 4, 52(1)
; AIX64-PWR7-NEXT:    stw 3, 52(1)
;  AIX64-COM-NEXT:    ld 1, 0(1)
;  AIX64-COM-NEXT:    blr
