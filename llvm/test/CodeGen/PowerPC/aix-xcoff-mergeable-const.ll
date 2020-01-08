; This file tests the codegen of mergeable const in AIX assembly.
; This file also tests mergeable const in XCOFF object file generation.
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=CHECKOBJ %s
; RUN: llvm-readobj -syms %t.o | FileCheck --check-prefix=CHECKSYM %s

%struct.Merge_cnst32 = type { i64, i32, i64, i32 }
%struct.Merge_cnst16 = type { i64, i32 }
%struct.Merge_cnst8 = type { i32, i32 }
%struct.Merge_cnst4 = type { i16, i8 }

@__const.main.cnst32 = private unnamed_addr constant %struct.Merge_cnst32 { i64 4611686018427387954, i32 0, i64 0, i32 0 }
@__const.main.cnst16 = private unnamed_addr constant %struct.Merge_cnst16 { i64 4611686018427387926, i32 0 }
@__const.main.cnst8 = private unnamed_addr constant %struct.Merge_cnst8 { i32 1073741832, i32 0 }
@__const.main.cnst4 = private unnamed_addr constant %struct.Merge_cnst4 { i16 16392, i8 0 }

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 {
entry:
  ret i32 0
}

;CHECK:         .csect .rodata[RO]
;CHECK-NEXT:         .align  4
;CHECK-NEXT: .L__const.main.cnst32:
;CHECK-NEXT:         .llong  4611686018427387954     # 0x4000000000000032
;CHECK-NEXT:         .long   0                       # 0x0
;CHECK-NEXT:         .space  4
;CHECK-NEXT:         .llong  0                       # 0x0
;CHECK-NEXT:         .long   0                       # 0x0
;CHECK-NEXT:         .space  4
;CHECK-NEXT:         .align  3
;CHECK-NEXT: .L__const.main.cnst16:
;CHECK-NEXT:         .llong  4611686018427387926     # 0x4000000000000016
;CHECK-NEXT:         .long   0                       # 0x0
;CHECK-NEXT:         .space  4
;CHECK-NEXT:         .align  3
;CHECK-NEXT: .L__const.main.cnst8:
;CHECK-NEXT:         .long   1073741832              # 0x40000008
;CHECK-NEXT:         .long   0                       # 0x0
;CHECK-NEXT:         .align  3
;CHECK-NEXT: .L__const.main.cnst4:
;CHECK-NEXT:         .short  16392                   # 0x4008
;CHECK-NEXT:         .byte   0                       # 0x0
;CHECK-NEXT:         .space  1


;CHECKOBJ:      00000000 .text:
;CHECKOBJ-NEXT:        0: 38 60 00 00                    li 3, 0
;CHECKOBJ-NEXT:        4: 4e 80 00 20                    blr
;CHECKOBJ-NEXT:          ...{{[[:space:]] *}}
;CHECKOBJ-NEXT: 00000010 .rodata:
;CHECKOBJ-NEXT:        10: 40 00 00 00                    bdnzf   0, .+0
;CHECKOBJ-NEXT:        14: 00 00 00 32                    <unknown>
;CHECKOBJ-NEXT:          ...{{[[:space:]] *}}
;CHECKOBJ-NEXT: 00000030 .L__const.main.cnst16:
;CHECKOBJ-NEXT:       30: 40 00 00 00                    bdnzf   0, .+0
;CHECKOBJ-NEXT:       34: 00 00 00 16                    <unknown>
;CHECKOBJ-NEXT:          ...{{[[:space:]] *}}
;CHECKOBJ-NEXT: 00000040 .L__const.main.cnst8:
;CHECKOBJ-NEXT:       40: 40 00 00 08                    bdnzf   0, .+8
;CHECKOBJ-NEXT:       44: 00 00 00 00                    <unknown>{{[[:space:]] *}}
;CHECKOBJ-NEXT: 00000048 .L__const.main.cnst4:
;CHECKOBJ-NEXT:       48: 40 08 00 00                    bdnzf   8, .+0


;CHECKSYM:        Symbol {{[{][[:space:]] *}}Index: [[#Index:]]{{[[:space:]] *}}Name: .rodata
;CHECKSYM-NEXT:     Value (RelocatableAddress): 0x10
;CHECKSYM-NEXT:     Section: .text
;CHECKSYM-NEXT:     Type: 0x0
;CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
;CHECKSYM-NEXT:     NumberOfAuxEntries: 1
;CHECKSYM-NEXT:     CSECT Auxiliary Entry {
;CHECKSYM-NEXT:       Index: [[#Index+1]]
;CHECKSYM-NEXT:       SectionLen: 60
;CHECKSYM-NEXT:       ParameterHashIndex: 0x0
;CHECKSYM-NEXT:       TypeChkSectNum: 0x0
;CHECKSYM-NEXT:       SymbolAlignmentLog2: 4
;CHECKSYM-NEXT:       SymbolType: XTY_SD (0x1)
;CHECKSYM-NEXT:       StorageMappingClass: XMC_RO (0x1)
;CHECKSYM-NEXT:       StabInfoIndex: 0x0
;CHECKSYM-NEXT:       StabSectNum: 0x0
;CHECKSYM-NEXT:     }
;CHECKSYM-NEXT:   }
;CHECKSYM-NEXT:   Symbol {
;CHECKSYM-NEXT:     Index: [[#Index+2]]
;CHECKSYM-NEXT:     Name: .L__const.main.cnst32
;CHECKSYM-NEXT:     Value (RelocatableAddress): 0x10
;CHECKSYM-NEXT:     Section: .text
;CHECKSYM-NEXT:     Type: 0x0
;CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
;CHECKSYM-NEXT:     NumberOfAuxEntries: 1
;CHECKSYM-NEXT:     CSECT Auxiliary Entry {
;CHECKSYM-NEXT:       Index: [[#Index+3]]
;CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index]]
;CHECKSYM-NEXT:       ParameterHashIndex: 0x0
;CHECKSYM-NEXT:       TypeChkSectNum: 0x0
;CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
;CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
;CHECKSYM-NEXT:       StorageMappingClass: XMC_RO (0x1)
;CHECKSYM-NEXT:       StabInfoIndex: 0x0
;CHECKSYM-NEXT:       StabSectNum: 0x0
;CHECKSYM-NEXT:     }
;CHECKSYM-NEXT:   }
;CHECKSYM-NEXT:   Symbol {
;CHECKSYM-NEXT:     Index: [[#Index+4]]
;CHECKSYM-NEXT:     Name: .L__const.main.cnst16
;CHECKSYM-NEXT:     Value (RelocatableAddress): 0x30
;CHECKSYM-NEXT:     Section: .text
;CHECKSYM-NEXT:     Type: 0x0
;CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
;CHECKSYM-NEXT:     NumberOfAuxEntries: 1
;CHECKSYM-NEXT:     CSECT Auxiliary Entry {
;CHECKSYM-NEXT:       Index: [[#Index+5]]
;CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index]]
;CHECKSYM-NEXT:       ParameterHashIndex: 0x0
;CHECKSYM-NEXT:       TypeChkSectNum: 0x0
;CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
;CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
;CHECKSYM-NEXT:       StorageMappingClass: XMC_RO (0x1)
;CHECKSYM-NEXT:       StabInfoIndex: 0x0
;CHECKSYM-NEXT:       StabSectNum: 0x0
;CHECKSYM-NEXT:     }
;CHECKSYM-NEXT:   }
;CHECKSYM-NEXT:   Symbol {
;CHECKSYM-NEXT:     Index: [[#Index+6]]
;CHECKSYM-NEXT:     Name: .L__const.main.cnst8
;CHECKSYM-NEXT:     Value (RelocatableAddress): 0x40
;CHECKSYM-NEXT:     Section: .text
;CHECKSYM-NEXT:     Type: 0x0
;CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
;CHECKSYM-NEXT:     NumberOfAuxEntries: 1
;CHECKSYM-NEXT:     CSECT Auxiliary Entry {
;CHECKSYM-NEXT:       Index: [[#Index+7]]
;CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index]]
;CHECKSYM-NEXT:       ParameterHashIndex: 0x0
;CHECKSYM-NEXT:       TypeChkSectNum: 0x0
;CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
;CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
;CHECKSYM-NEXT:       StorageMappingClass: XMC_RO (0x1)
;CHECKSYM-NEXT:       StabInfoIndex: 0x0
;CHECKSYM-NEXT:       StabSectNum: 0x0
;CHECKSYM-NEXT:     }
;CHECKSYM-NEXT:   }
;CHECKSYM-NEXT:   Symbol {
;CHECKSYM-NEXT:     Index: [[#Index+8]]
;CHECKSYM-NEXT:     Name: .L__const.main.cnst4
;CHECKSYM-NEXT:     Value (RelocatableAddress): 0x48
;CHECKSYM-NEXT:     Section: .text
;CHECKSYM-NEXT:     Type: 0x0
;CHECKSYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
;CHECKSYM-NEXT:     NumberOfAuxEntries: 1
;CHECKSYM-NEXT:     CSECT Auxiliary Entry {
;CHECKSYM-NEXT:       Index: [[#Index+9]]
;CHECKSYM-NEXT:       ContainingCsectSymbolIndex: [[#Index]]
;CHECKSYM-NEXT:       ParameterHashIndex: 0x0
;CHECKSYM-NEXT:       TypeChkSectNum: 0x0
;CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
;CHECKSYM-NEXT:       SymbolType: XTY_LD (0x2)
;CHECKSYM-NEXT:       StorageMappingClass: XMC_RO (0x1)
;CHECKSYM-NEXT:       StabInfoIndex: 0x0
;CHECKSYM-NEXT:       StabSectNum: 0x0
;CHECKSYM-NEXT:     }
;CHECKSYM-NEXT:   }
