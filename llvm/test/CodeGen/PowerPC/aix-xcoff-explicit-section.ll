; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D --symbol-description %t.o | FileCheck --check-prefix=CHECKOBJ %s
; RUN: llvm-readobj -syms %t.o | FileCheck --check-prefix=CHECKSYM %s

@ext_const = constant i32 1, section ".ext_const_sec", align 4
@ext_var = global i32 1, section ".ext_var_sec", align 4
@ext_zvar = global i32 0, section ".ext_zvar_sec", align 4

define dso_local signext i32 @ext_fun() section ".ext_fun_sec" {
entry:
  %0 = load i32, i32* @ext_const, align 4
  %1 = load i32, i32* @ext_var, align 4
  %add = add nsw i32 %0, %1
  %2 = load i32, i32* @ext_zvar, align 4
  %add1 = add nsw i32 %add, %2
  ret i32 %add1
}

; CHECK:              .globl  ext_fun[DS]                     # -- Begin function ext_fun
; CHECK-NEXT:         .globl  .ext_fun
; CHECK-NEXT:         .align  4
; CHECK-NEXT:         .csect ext_fun[DS]
; CHECK:              .csect .ext_fun_sec[PR],2
; CHECK-NEXT: .ext_fun:
; CHECK:              .csect .ext_const_sec[RO],2
; CHECK-NEXT:         .globl  ext_const
; CHECK-NEXT:         .align  2
; CHECK-NEXT: ext_const:
; CHECK-NEXT:         .vbyte  4, 1                            # 0x1
; CHECK-NEXT:         .csect .ext_var_sec[RW],2
; CHECK-NEXT:         .globl  ext_var
; CHECK-NEXT:         .align  2
; CHECK-NEXT: ext_var:
; CHECK-NEXT:         .vbyte  4, 1                            # 0x1
; CHECK-NEXT:         .csect .ext_zvar_sec[RW],2
; CHECK-NEXT:         .globl  ext_zvar
; CHECK-NEXT:         .align  2
; CHECK-NEXT: ext_zvar:
; CHECK-NEXT:         .vbyte  4, 0                            # 0x0
; CHECK-NEXT:         .toc
; CHECK-NEXT: L..C0:
; CHECK-NEXT:         .tc ext_var[TC],ext_var
; CHECK-NEXT: L..C1:
; CHECK-NEXT:         .tc ext_zvar[TC],ext_zvar

; CHECKOBJ:        00000000 (idx: 5) .ext_fun:
; CHECKOBJ-NEXT:          0: 80 62 00 00   lwz 3, 0(2)
; CHECKOBJ-NEXT:          4: 80 82 00 04   lwz 4, 4(2)
; CHECKOBJ-NEXT:          8: 80 63 00 00   lwz 3, 0(3)
; CHECKOBJ-NEXT:          c: 80 84 00 00   lwz 4, 0(4)
; CHECKOBJ-NEXT:         10: 7c 63 22 14   add 3, 3, 4
; CHECKOBJ-NEXT:         14: 38 63 00 01   addi 3, 3, 1
; CHECKOBJ-NEXT:         18: 4e 80 00 20   blr
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   0000001c (idx: 9) ext_const:
; CHECKOBJ-NEXT:         1c: 00 00 00 01   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   Disassembly of section .data:
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000020 (idx: 13) ext_var:
; CHECKOBJ-NEXT:         20: 00 00 00 01   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000024 (idx: 17) ext_zvar:
; CHECKOBJ-NEXT:         24: 00 00 00 00   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000028 (idx: 19) ext_fun[DS]:
; CHECKOBJ-NEXT:         28: 00 00 00 00   <unknown>
; CHECKOBJ-NEXT:         2c: 00 00 00 34   <unknown>
; CHECKOBJ-NEXT:         30: 00 00 00 00   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000034 (idx: 23) ext_var[TC]:
; CHECKOBJ-NEXT:         34: 00 00 00 20   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000038 (idx: 25) ext_zvar[TC]:
; CHECKOBJ-NEXT:         38: 00 00 00 24   <unknown>

; CHECKSYM:       Symbol {{[{][[:space:]] *}}Index: [[#INDX:]]{{[[:space:]] *}}Name: .ext_fun_sec
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:      Section: .text
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+1]]
; CHECKSYM-NEXT:        SectionLen: 28
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 4
; CHECKSYM-NEXT:        SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_PR (0x0)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
; CHECKSYM-NEXT:    Symbol {
; CHECKSYM-NEXT:      Index: [[#INDX+2]]
; CHECKSYM-NEXT:      Name: .ext_fun
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:      Section: .text
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+3]]
; CHECKSYM-NEXT:        ContainingCsectSymbolIndex: [[#INDX]]
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:        SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_PR (0x0)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
; CHECKSYM-NEXT:    Symbol {
; CHECKSYM-NEXT:      Index: [[#INDX+4]]
; CHECKSYM-NEXT:      Name: .ext_const_sec
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x1C
; CHECKSYM-NEXT:      Section: .text
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+5]]
; CHECKSYM-NEXT:        SectionLen: 4
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:        SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_RO (0x1)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
; CHECKSYM-NEXT:    Symbol {
; CHECKSYM-NEXT:      Index: [[#INDX+6]]
; CHECKSYM-NEXT:      Name: ext_const
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x1C
; CHECKSYM-NEXT:      Section: .text
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+7]]
; CHECKSYM-NEXT:        ContainingCsectSymbolIndex: [[#INDX+4]]
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:        SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_RO (0x1)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
; CHECKSYM-NEXT:    Symbol {
; CHECKSYM-NEXT:      Index: [[#INDX+8]]
; CHECKSYM-NEXT:      Name: .ext_var_sec
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x20
; CHECKSYM-NEXT:      Section: .data
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+9]]
; CHECKSYM-NEXT:        SectionLen: 4
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:        SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_RW (0x5)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
; CHECKSYM-NEXT:    Symbol {
; CHECKSYM-NEXT:      Index: [[#INDX+10]]
; CHECKSYM-NEXT:      Name: ext_var
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x20
; CHECKSYM-NEXT:      Section: .data
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+11]]
; CHECKSYM-NEXT:        ContainingCsectSymbolIndex: [[#INDX+8]]
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:        SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_RW (0x5)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
; CHECKSYM-NEXT:    Symbol {
; CHECKSYM-NEXT:      Index: [[#INDX+12]]
; CHECKSYM-NEXT:      Name: .ext_zvar_sec
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x24
; CHECKSYM-NEXT:      Section: .data
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+13]]
; CHECKSYM-NEXT:        SectionLen: 4
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:        SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_RW (0x5)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
; CHECKSYM-NEXT:    Symbol {
; CHECKSYM-NEXT:      Index: [[#INDX+14]]
; CHECKSYM-NEXT:      Name: ext_zvar
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x24
; CHECKSYM-NEXT:      Section: .data
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+15]]
; CHECKSYM-NEXT:        ContainingCsectSymbolIndex: [[#INDX+12]]
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:        SymbolType: XTY_LD (0x2)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_RW (0x5)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
; CHECKSYM-NEXT:    Symbol {
; CHECKSYM-NEXT:      Index: [[#INDX+16]]
; CHECKSYM-NEXT:      Name: ext_fun
; CHECKSYM-NEXT:      Value (RelocatableAddress): 0x28
; CHECKSYM-NEXT:      Section: .data
; CHECKSYM-NEXT:      Type: 0x0
; CHECKSYM-NEXT:      StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:      NumberOfAuxEntries: 1
; CHECKSYM-NEXT:      CSECT Auxiliary Entry {
; CHECKSYM-NEXT:        Index: [[#INDX+17]]
; CHECKSYM-NEXT:        SectionLen: 12
; CHECKSYM-NEXT:        ParameterHashIndex: 0x0
; CHECKSYM-NEXT:        TypeChkSectNum: 0x0
; CHECKSYM-NEXT:        SymbolAlignmentLog2: 2
; CHECKSYM-NEXT:        SymbolType: XTY_SD (0x1)
; CHECKSYM-NEXT:        StorageMappingClass: XMC_DS (0xA)
; CHECKSYM-NEXT:        StabInfoIndex: 0x0
; CHECKSYM-NEXT:        StabSectNum: 0x0
; CHECKSYM-NEXT:      }
; CHECKSYM-NEXT:    }
