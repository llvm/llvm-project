; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -data-sections -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff -data-sections < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -filetype=obj -data-sections -xcoff-traceback-table=false -o %t.o < %s
; RUN: llvm-objdump -D --symbol-description %t.o | FileCheck --check-prefix=CHECKOBJ %s
; RUN: llvm-readobj -syms %t.o | FileCheck --check-prefix=CHECKSYM %s

;; Test to see if the default is correct for -data-sections on AIX.
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D --symbol-description %t.o | FileCheck --check-prefix=CHECKOBJ %s
; RUN: llvm-readobj -syms %t.o | FileCheck --check-prefix=CHECKSYM %s

;; Test to see if the default is correct for -data-sections on AIX.
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D --symbol-description %t.o | FileCheck --check-prefix=CHECKOBJ %s
; RUN: llvm-readobj -syms %t.o | FileCheck --check-prefix=CHECKSYM %s

@ivar = local_unnamed_addr global i32 35, align 4
@const_ivar = constant i32 35, align 4

@a = common global i32 0, align 4
@f = common local_unnamed_addr global i32 0, align 4

@.str = private unnamed_addr constant [9 x i8] c"abcdefgh\00", align 1
@p = global i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0), align 4

define i8 @foo() {
entry:
  %0 = load i8*, i8** @p, align 4
  %1 = load i8, i8* %0, align 1
  ret i8 %1
}

define i32 @bar() {
entry:
  %0 = load i32, i32* @ivar, align 4
  %1 = load i32, i32* @const_ivar, align 4
  %add = add nsw i32 %0, %1
  %2 = load i32, i32* @a, align 4
  %add1 = add nsw i32 %add, %2
  %3 = load i32, i32* @f, align 4
  %add2 = add nsw i32 %add1, %3
  ret i32 %add2
}


; CHECK:              .csect ivar[RW],2
; CHECK-NEXT:         .globl  ivar[RW]
; CHECK-NEXT:         .align  2
; CHECK-NEXT:         .vbyte  4, 35                           # 0x23
; CHECK-NEXT:         .csect const_ivar[RO],2
; CHECK-NEXT:         .globl  const_ivar[RO]
; CHECK-NEXT:         .align  2
; CHECK-NEXT:         .vbyte  4, 35                           # 0x23
; CHECK-NEXT:         .comm   a[RW],4,2
; CHECK-NEXT:         .comm   f[RW],4,2
; CHECK-NEXT:         .csect .rodata.str1.1L...str[RO],2
; CHECK-NEXT:         .byte 'a,'b,'c,'d,'e,'f,'g,'h,0000
; CHECK32:            .csect p[RW],2
; CHECK32-NEXT:       .globl  p[RW]
; CHECK32-NEXT:       .align  2
; CHECK32-NEXT:       .vbyte  4, .rodata.str1.1L...str[RO]
; CHECK64:            .csect p[RW],3
; CHECK64-NEXT:       .globl  p[RW]
; CHECK64-NEXT:       .align  3
; CHECK64-NEXT:       .vbyte  8, .rodata.str1.1L...str[RO]
; CHECK:              .toc
; CHECK-NEXT: L..C0:
; CHECK-NEXT:         .tc p[TC],p[RW]
; CHECK-NEXT: L..C1:
; CHECK-NEXT:         .tc ivar[TC],ivar[RW]
; CHECK-NEXT: L..C2:
; CHECK-NEXT:         .tc a[TC],a[RW]
; CHECK-NEXT: L..C3:
; CHECK-NEXT:         .tc f[TC],f[RW]

; CHECKOBJ:        00000038 (idx: 6) const_ivar[RO]:
; CHECKOBJ-NEXT:         38: 00 00 00 23   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   0000003c (idx: 8) .rodata.str1.1L...str[RO]:
; CHECKOBJ-NEXT:         3c: 61 62 63 64
; CHECKOBJ-NEXT:         40: 65 66 67 68
; CHECKOBJ-NEXT:         44: 00 00 00 00   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   Disassembly of section .data:
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000048 (idx: 10) ivar[RW]:
; CHECKOBJ-NEXT:         48: 00 00 00 23   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   0000004c (idx: 12) p[RW]:
; CHECKOBJ-NEXT:         4c: 00 00 00 3c   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000050 (idx: 14) foo[DS]:
; CHECKOBJ-NEXT:         50: 00 00 00 00   <unknown>
; CHECKOBJ-NEXT:         54: 00 00 00 68   <unknown>
; CHECKOBJ-NEXT:         58: 00 00 00 00   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   0000005c (idx: 16) bar[DS]:
; CHECKOBJ-NEXT:         5c: 00 00 00 10   <unknown>
; CHECKOBJ-NEXT:         60: 00 00 00 68   <unknown>
; CHECKOBJ-NEXT:         64: 00 00 00 00   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000068 (idx: 20) p[TC]:
; CHECKOBJ-NEXT:         68: 00 00 00 4c   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   0000006c (idx: 22) ivar[TC]:
; CHECKOBJ-NEXT:         6c: 00 00 00 48   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000070 (idx: 24) a[TC]:
; CHECKOBJ-NEXT:         70: 00 00 00 78   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000074 (idx: 26) f[TC]:
; CHECKOBJ-NEXT:         74: 00 00 00 7c   <unknown>
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   Disassembly of section .bss:
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   00000078 (idx: 28) a[RW]:
; CHECKOBJ-NEXT:   ...
; CHECKOBJ-EMPTY:
; CHECKOBJ-NEXT:   0000007c (idx: 30) f[RW]:
; CHECKOBJ-NEXT:   ...


; CHECKSYM:    Symbol {
; CHECKSYM:      Name: const_ivar
; CHECKSYM:      Value (RelocatableAddress): 0x38
; CHECKSYM:      Section: .text
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_EXT (0x2)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_RO (0x1)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: .rodata.str1.1L...str
; CHECKSYM:      Value (RelocatableAddress): 0x3C
; CHECKSYM:      Section: .text
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 9
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_RO (0x1)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: ivar
; CHECKSYM:      Value (RelocatableAddress): 0x48
; CHECKSYM:      Section: .data
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_EXT (0x2)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_RW (0x5)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: p
; CHECKSYM:      Value (RelocatableAddress): 0x4C
; CHECKSYM:      Section: .data
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_EXT (0x2)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_RW (0x5)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: TOC
; CHECKSYM:      Value (RelocatableAddress): 0x68
; CHECKSYM:      Section: .data
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 0
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_TC0 (0xF)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: p
; CHECKSYM:      Value (RelocatableAddress): 0x68
; CHECKSYM:      Section: .data
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_TC (0x3)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: ivar
; CHECKSYM:      Value (RelocatableAddress): 0x6C
; CHECKSYM:      Section: .data
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_TC (0x3)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: a
; CHECKSYM:      Value (RelocatableAddress): 0x70
; CHECKSYM:      Section: .data
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_TC (0x3)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: f
; CHECKSYM:      Value (RelocatableAddress): 0x74
; CHECKSYM:      Section: .data
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_HIDEXT (0x6B)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_SD (0x1)
; CHECKSYM:        StorageMappingClass: XMC_TC (0x3)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: a
; CHECKSYM:      Value (RelocatableAddress): 0x78
; CHECKSYM:      Section: .bss
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_EXT (0x2)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_CM (0x3)
; CHECKSYM:        StorageMappingClass: XMC_RW (0x5)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
; CHECKSYM:    Symbol {
; CHECKSYM:      Name: f
; CHECKSYM:      Value (RelocatableAddress): 0x7C
; CHECKSYM:      Section: .bss
; CHECKSYM:      Type: 0x0
; CHECKSYM:      StorageClass: C_EXT (0x2)
; CHECKSYM:      NumberOfAuxEntries: 1
; CHECKSYM:      CSECT Auxiliary Entry {
; CHECKSYM:        SectionLen: 4
; CHECKSYM:        ParameterHashIndex: 0x0
; CHECKSYM:        TypeChkSectNum: 0x0
; CHECKSYM:        SymbolAlignmentLog2: 2
; CHECKSYM:        SymbolType: XTY_CM (0x3)
; CHECKSYM:        StorageMappingClass: XMC_RW (0x5)
; CHECKSYM:        StabInfoIndex: 0x0
; CHECKSYM:        StabSectNum: 0x0
; CHECKSYM:      }
; CHECKSYM:    }
