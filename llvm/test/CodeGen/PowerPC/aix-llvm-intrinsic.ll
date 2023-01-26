; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefixes=CHECKSYM,CHECKSYM32 %s
; RUN: llvm-objdump -r -d --symbol-description %t.o | FileCheck --check-prefixes=CHECKRELOC,CHECKRELOC32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck --check-prefixes=CHECKSYM,CHECKSYM64 %s
; RUN: llvm-objdump -r -d --symbol-description %t64.o | FileCheck --check-prefixes=CHECKRELOC,CHECKRELOC64 %s

%struct.S = type { i32, i32 }

@s = external global %struct.S, align 4

define void @bar() {
entry:
  %0 = load i32, ptr getelementptr inbounds (%struct.S, ptr @s, i32 0, i32 1), align 4
  %1 = trunc i32 %0 to i8
  %2 = load i32, ptr getelementptr inbounds (%struct.S, ptr @s, i32 0, i32 1), align 4
  call void @llvm.memset.p0.i32(ptr align 4 @s, i8 %1, i32 %2, i1 false)
  ret void
}

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)

; CHECK-LABEL: .bar:
; CHECK-NEXT: # %bb.0:                                # %entry
; CHECK-NEXT:         mflr 0

; CHECK:              bl .memset

; CHECK:              .extern .memset

; CHECKSYM:        Symbol {
; CHECKSYM-NEXT:     Index: 0
; CHECKSYM-NEXT:     Name: <stdin>
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
; CHECKSYM32-NEXT:     StabInfoIndex: 0x0
; CHECKSYM32-NEXT:     StabSectNum: 0x0
; CHECKSYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; CHECKSYM-NEXT:     }
; CHECKSYM-NEXT:   }

; CHECKRELOC32:      00000000 (idx: 7) .bar:
; CHECKRELOC64:      0000000000000000 (idx: 7) .bar:
; CHECKRELOC-NEXT:        0: 7c 08 02 a6                        mflr 0
; CHECKRELOC32-NEXT:        4: 94 21 ff c0                      stwu 1, -64(1)
; CHECKRELOC32-NEXT:        8: 80 62 00 00                      lwz 3, 0(2)
; CHECKRELOC32-NEXT:    0000000a:  R_TOC        (idx: 13) s[TC]
; CHECKRELOC32-NEXT:        c: 90 01 00 48                      stw 0, 72(1)
; CHECKRELOC64-NEXT:        4: f8 21 ff 91                      stdu 1, -112(1)
; CHECKRELOC64-NEXT:        8: e8 62 00 00                      ld 3, 0(2)
; CHECKRELOC64-NEXT:    000000000000000a:  R_TOC	(idx: 13) s[TC]
; CHECKRELOC64-NEXT:        c: f8 01 00 80                      std 0, 128(1)
; CHECKRELOC-NEXT:       10: 80 83 00 04                        lwz 4, 4(3)
; CHECKRELOC-NEXT:       14: 7c 85 23 78                        mr 5, 4
; CHECKRELOC-NEXT:       18: 4b ff ff e9                        bl 0x0
; CHECKRELOC32-NEXT:    00000018:  R_RBR        (idx: 1) .memset[PR]
; CHECKRELOC64-NEXT:    0000000000000018:  R_RBR	(idx: 1) .memset[PR]
