; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefix=CHECKSYM %s
; RUN: llvm-objdump -r -d --symbol-description %t.o | FileCheck --check-prefix=CHECKRELOC %s

; RUN: not --crash llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff \
; RUN:                 -mattr=-altivec -filetype=obj -o %t.o 2>&1 < %s | \
; RUN:   FileCheck --check-prefix=XCOFF64 %s
; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

%struct.S = type { i32, i32 }

@s = external global %struct.S, align 4

define void @bar() {
entry:
  %0 = load i32, i32* getelementptr inbounds (%struct.S, %struct.S* @s, i32 0, i32 1), align 4
  %1 = trunc i32 %0 to i8
  %2 = load i32, i32* getelementptr inbounds (%struct.S, %struct.S* @s, i32 0, i32 1), align 4
  call void @llvm.memset.p0i8.i32(i8* align 4 bitcast (%struct.S* @s to i8*), i8 %1, i32 %2, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1 immarg)

; CHECK-LABEL: .bar:
; CHECK-NEXT: # %bb.0:                                # %entry
; CHECK-NEXT:         mflr 0

; CHECK:              bl .memset

; CHECK:              .extern .memset

; CHECKSYM:        Symbol {
; CHECKSYM-NEXT:     Index: 0
; CHECKSYM-NEXT:     Name: .file
; CHECKSYM-NEXT:     Value (SymbolTableIndex): 0x0
; CHECKSYM-NEXT:     Section: N_DEBUG
; CHECKSYM-NEXT:     Source Language ID: TB_C (0x0)
; CHECKSYM-NEXT:     CPU Version ID: 0x0
; CHECKSYM-NEXT:     StorageClass: C_FILE (0x67)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 0
; CHECKSYM-NEXT:   }
; CHECKSYM-NEXT:   Symbol {
; CHECKSYM-NEXT:     Index: 1
; CHECKSYM-NEXT:     Name: .memset
; CHECKSYM-NEXT:     Value (RelocatableAddress): 0x0
; CHECKSYM-NEXT:     Section: N_UNDEF
; CHECKSYM-NEXT:     Type: 0x0
; CHECKSYM-NEXT:     StorageClass: C_EXT (0x2)
; CHECKSYM-NEXT:     NumberOfAuxEntries: 1
; CHECKSYM-NEXT:     CSECT Auxiliary Entry {
; CHECKSYM-NEXT:       Index: 2
; CHECKSYM-NEXT:       SectionLen: 0
; CHECKSYM-NEXT:       ParameterHashIndex: 0x0
; CHECKSYM-NEXT:       TypeChkSectNum: 0x0
; CHECKSYM-NEXT:       SymbolAlignmentLog2: 0
; CHECKSYM-NEXT:       SymbolType: XTY_ER (0x0)
; CHECKSYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECKSYM-NEXT:       StabInfoIndex: 0x0
; CHECKSYM-NEXT:       StabSectNum: 0x0
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }

; CHECKRELOC:      00000000 (idx: 7) .bar:
; CHECKRELOC-NEXT:        0: 7c 08 02 a6                        mflr 0
; CHECKRELOC-NEXT:        4: 90 01 00 08                        stw 0, 8(1)
; CHECKRELOC-NEXT:        8: 94 21 ff c0                        stwu 1, -64(1)
; CHECKRELOC-NEXT:        c: 80 62 00 00                        lwz 3, 0(2)
; CHECKRELOC-NEXT:                      0000000e:  R_TOC        (idx: 13) s[TC]
; CHECKRELOC-NEXT:       10: 80 83 00 04                        lwz 4, 4(3)
; CHECKRELOC-NEXT:       14: 7c 85 23 78                        mr 5, 4
; CHECKRELOC-NEXT:       18: 4b ff ff e9                        bl 0x0
; CHECKRELOC-NEXT:                      00000018:  R_RBR        (idx: 1) .memset[PR]
