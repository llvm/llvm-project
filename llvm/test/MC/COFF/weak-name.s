// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -defsym case=0 -o %t.case0.o
// RUN: llvm-readobj --symbols %t.case0.o | FileCheck %s --check-prefix=CHECK-CASE0
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -defsym case=1 -o %t.case1.o
// RUN: llvm-readobj --symbols %t.case1.o | FileCheck %s --check-prefix=CHECK-CASE1
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -defsym case=2 -o %t.case2.o
// RUN: llvm-readobj --symbols %t.case2.o | FileCheck %s --check-prefix=CHECK-CASE2

// Test that we prefer a non-comdat symbol for naming weak default symbols,
// if such a symbol is available.

        .section .text$comdat1,"xr",discard,comdat1
        .globl   comdat1
comdat1:
        call     undeffunc

        .weak    weaksym

        .section .text$comdat2,"xr",discard,comdat2
        .globl   comdat2
comdat2:
        call     undeffunc2

.if case == 0
        .text
        .globl   regular
regular:
        call     undeffunc3
.elseif case == 1
        .globl   abssym
abssym = 42
.endif

// CHECK-CASE0: Name: .weak.weaksym.default.regular
// CHECK-CASE1: Name: .weak.weaksym.default.abssym
// CHECK-CASE2: Name: .weak.weaksym.default.comdat1
