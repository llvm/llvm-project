; RUN: llc -mtriple=x86_64-linux-gnu < %s | FileCheck -check-prefix=LINUX %s
; RUN: not llc -mtriple=x86_64-apple-macos10.9 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-ios9.0 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-tvos9.0 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-watchos9.0 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-xros9.0 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-ios8.0 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-tvos8.0 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-xros8.0 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-driverkit < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=x86_64-apple-driverkit24.0 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: llc < %s -mtriple=i686-linux-gnu -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=GISEL-X86
; RUN: llc < %s -mtriple=x86_64-linux-gnu -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=GISEL-X64

; ERR: no libcall available for fexp10

define x86_fp80 @test_exp10_f80(x86_fp80 %x) nounwind {
; LINUX-LABEL: test_exp10_f80:
; LINUX:       # %bb.0:
; LINUX-NEXT:    subq $24, %rsp
; LINUX-NEXT:    fldt {{[0-9]+}}(%rsp)
; LINUX-NEXT:    fstpt (%rsp)
; LINUX-NEXT:    callq exp10l@PLT
; LINUX-NEXT:    addq $24, %rsp
; LINUX-NEXT:    retq
;
; GISEL-X86-LABEL: test_exp10_f80:
; GISEL-X86:       # %bb.0:
; GISEL-X86-NEXT:    subl $12, %esp
; GISEL-X86-NEXT:    fldt {{[0-9]+}}(%esp)
; GISEL-X86-NEXT:    fstpt (%esp)
; GISEL-X86-NEXT:    calll exp10l
; GISEL-X86-NEXT:    addl $12, %esp
; GISEL-X86-NEXT:    retl
;
; GISEL-X64-LABEL: test_exp10_f80:
; GISEL-X64:       # %bb.0:
; GISEL-X64-NEXT:    subq $24, %rsp
; GISEL-X64-NEXT:    fldt {{[0-9]+}}(%rsp)
; GISEL-X64-NEXT:    fstpt (%rsp)
; GISEL-X64-NEXT:    callq exp10l
; GISEL-X64-NEXT:    addq $24, %rsp
; GISEL-X64-NEXT:    retq
  %ret = call x86_fp80 @llvm.exp10.f80(x86_fp80 %x)
  ret x86_fp80 %ret
}
