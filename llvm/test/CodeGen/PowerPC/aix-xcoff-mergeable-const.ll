; This file tests the codegen of mergeable const in AIX assembly.
; This file also tests mergeable const in XCOFF object file generation.
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -global-merge-all-const=false -data-sections=false -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -global-merge-all-const=false -xcoff-traceback-table=false -data-sections=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -global-merge-all-const=false -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=CHECKOBJ %s
; RUN: llvm-readobj -s %t.o | FileCheck --check-prefix=CHECKSYM %s

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

;CHECK:         .csect .rodata[RO],4

;CHECK-NEXT:         .align  4
;CHECK-NEXT: L..__const.main.cnst32:
;CHECK32-NEXT:         .vbyte	4, 1073741824
;CHECK32-NEXT:         .vbyte	4, 50
;CHECK64-NEXT:         .vbyte	8, 4611686018427387954
;CHECK-NEXT:           .vbyte	4, 0                       # 0x0
;CHECK-NEXT:           .space  4
;CHECK32-NEXT:         .vbyte	4, 0
;CHECK32-NEXT:         .vbyte	4, 0
;CHECK64-NEXT:         .vbyte	8, 0
;CHECK-NEXT:           .vbyte	4, 0                       # 0x0
;CHECK-NEXT:           .space  4

;CHECK-NEXT:           .align  3
;CHECK-NEXT: L..__const.main.cnst16:
;CHECK32-NEXT:         .vbyte	4, 1073741824
;CHECK32-NEXT:         .vbyte	4, 22
;CHECK64-NEXT:         .vbyte	8, 4611686018427387926
;CHECK-NEXT:           .vbyte	4, 0                       # 0x0
;CHECK-NEXT:           .space  4

;CHECK-NEXT:         .align  3
;CHECK-NEXT: L..__const.main.cnst8:
;CHECK-NEXT:         .vbyte	4, 1073741832              # 0x40000008
;CHECK-NEXT:         .vbyte	4, 0                       # 0x0

;CHECK-NEXT:         .align  3
;CHECK-NEXT: L..__const.main.cnst4:
;CHECK-NEXT:         .vbyte	2, 16392                   # 0x4008
;CHECK-NEXT:         .byte   0                       # 0x0
;CHECK-NEXT:         .space  1


;CHECKOBJ:      00000000 <.main>:
;CHECKOBJ-NEXT:        0: 38 60 00 00                    li 3, 0
;CHECKOBJ-NEXT:        4: 4e 80 00 20                    blr
;CHECKOBJ-NEXT:          ...{{[[:space:]] *}}
;CHECKOBJ-NEXT: 00000010 <.rodata>:
;CHECKOBJ-NEXT:        10: 40 00 00 00
;CHECKOBJ-NEXT:        14: 00 00 00 32
;CHECKOBJ-NEXT:          ...{{[[:space:]] *}}
;CHECKOBJ-SAME:       30: 40 00 00 00
;CHECKOBJ-NEXT:       34: 00 00 00 16
;CHECKOBJ-NEXT:          ...{{[[:space:]] *}}
;CHECKOBJ-SAME:       40: 40 00 00 08
;CHECKOBJ-NEXT:       44: 00 00 00 00
;CHECKOBJ-NEXT:       48: 40 08 00 00


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
