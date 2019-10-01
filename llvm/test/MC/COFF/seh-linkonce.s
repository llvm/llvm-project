// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj --sections --section-symbols | FileCheck %s

        .text
        .def     weak_func;
        .scl    2;
        .type   32;
        .endef
        .section        .text,"xr",discard,weak_func
        .globl  weak_func
        .align  16, 0x90
weak_func:                              # @weak_func
.Ltmp0:
.seh_proc weak_func
# %bb.0:                                # %entry
        pushq   %rbp
.Ltmp1:
        .seh_pushreg 5
        movq    %rsp, %rbp
.Ltmp2:
        .seh_setframe 5, 0
.Ltmp3:
        .seh_endprologue
        xorl    %eax, %eax
        popq    %rbp
        retq
.Leh_func_end0:
.Ltmp4:
        .seh_endproc

// CHECK: Sections [
// CHECK:   Section {
// CHECK:     Name: .text
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: .data
// CHECK:   }
// CHECK:   Section {
// CHECK:     Name: .bss
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: [[TEXT_SECNUM:[0-9]+]]
// CHECK:     Name: .text
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: [[XDATA_SECNUM:[0-9]+]]
// CHECK:     Name: .xdata
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:     Symbols [
// CHECK:       Symbol {
// CHECK:         Name: .xdata
// CHECK:         Section: .xdata ([[XDATA_SECNUM]])
// CHECK:         StorageClass: Static (0x3)
// CHECK:         AuxSymbolCount: 1
// CHECK:         AuxSectionDef {
// CHECK:           Selection: Associative (0x5)
// CHECK:           AssocSection: .text ([[TEXT_SECNUM]])
// CHECK:         }
// CHECK:       }
// CHECK:     ]
// CHECK:   }
// CHECK:   Section {
// CHECK:     Number: [[PDATA_SECNUM:[0-9]+]]
// CHECK:     Name: .pdata
// CHECK:     Characteristics [
// CHECK:       IMAGE_SCN_LNK_COMDAT
// CHECK:     ]
// CHECK:     Symbols [
// CHECK:       Symbol {
// CHECK:         Name: .pdata
// CHECK:         Section: .pdata ([[PDATA_SECNUM]])
// CHECK:         StorageClass: Static (0x3)
// CHECK:         AuxSymbolCount: 1
// CHECK:         AuxSectionDef {
// CHECK:           Selection: Associative (0x5)
// CHECK:           AssocSection: .text ([[TEXT_SECNUM]])
// CHECK:         }
// CHECK:       }
// CHECK:     ]
// CHECK:   }
// CHECK: ]
