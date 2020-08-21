; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec < %s | \
; RUN:   FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec < %s |\
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | \
; RUN:   FileCheck --check-prefix=XCOFF32 %s

; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -mcpu=pwr4 -mattr=-altivec -filetype=obj -o %t.o 2>&1 < %s | \
; RUN:   FileCheck --check-prefix=XCOFF64 %s
; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

@b =  global i32 0, align 4
@b_h = hidden global i32 0, align 4

define void @foo() {
entry:
  ret void
}

define hidden void @foo_h(i32* %ip) {
entry:
  ret void
}

define protected void @foo_protected(i32* %ip) {
entry:
  ret void
}

define weak hidden void @foo_weak_h() {
entry:
  ret void
}

@foo_p = global void ()* @zoo_weak_extern_h, align 4
declare extern_weak hidden void @zoo_weak_extern_h()

define i32 @main() {
entry:
  %call1= call i32 @bar_h(i32* @b_h)
  call void @foo_weak_h()
  %0 = load void ()*, void ()** @foo_p, align 4
  call void %0()
  ret i32 0
}

declare hidden i32 @bar_h(i32*)

; CHECK:        .globl  foo[DS]{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  .foo{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  foo_h[DS],hidden
; CHECK:        .globl  .foo_h,hidden
; CHECK:        .globl  foo_protected[DS],protected
; CHECK:        .globl  .foo_protected,protected
; CHECK:        .weak   foo_weak_h[DS],hidden
; CHECK:        .weak   .foo_weak_h,hidden

; CHECK:        .globl  b{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  b_h,hidden

; CHECK:        .weak   .zoo_weak_extern_h[PR],hidden
; CHECK:        .weak   zoo_weak_extern_h[DS],hidden
; CHECK:        .extern .bar_h[PR],hidden
; CHECK:        .extern bar_h[DS],hidden

; XCOFF32:       Symbols [
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index:]]
; XCOFF32-NEXT:     Name: .bar_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:     Section: N_UNDEF
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+1]]
; XCOFF32-NEXT:       SectionLen: 0
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_ER (0x0)
; XCOFF32-NEXT:       StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+2]]
; XCOFF32-NEXT:     Name: zoo_weak_extern_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:     Section: N_UNDEF
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+3]]
; XCOFF32-NEXT:       SectionLen: 0
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_ER (0x0)
; XCOFF32-NEXT:       StorageMappingClass: XMC_DS (0xA)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+4]]
; XCOFF32-NEXT:     Name: .zoo_weak_extern_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:     Section: N_UNDEF
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+5]]
; XCOFF32-NEXT:       SectionLen: 0
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_ER (0x0)
; XCOFF32-NEXT:       StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+6]]
; XCOFF32-NEXT:     Name: bar_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:     Section: N_UNDEF
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+7]]
; XCOFF32-NEXT:       SectionLen: 0
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_ER (0x0)
; XCOFF32-NEXT:       StorageMappingClass: XMC_DS (0xA)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+8]]
; XCOFF32-NEXT:     Name: .text
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:     Section: .text
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+9]]
; XCOFF32-NEXT:       SectionLen: 152
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 4
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+10]]
; XCOFF32-NEXT:     Name: .foo
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:     Section: .text
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+11]]
; XCOFF32-NEXT:       ContainingCsectSymbolIndex: 8
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_LD (0x2)
; XCOFF32-NEXT:       StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+12]]
; XCOFF32-NEXT:     Name: .foo_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x10
; XCOFF32-NEXT:     Section: .text
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+13]]
; XCOFF32-NEXT:       ContainingCsectSymbolIndex: 8
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_LD (0x2)
; XCOFF32-NEXT:       StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+14]]
; XCOFF32-NEXT:     Name: .foo_protected
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x20
; XCOFF32-NEXT:     Section: .text
; XCOFF32-NEXT:     Type: 0x3000
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+15]]
; XCOFF32-NEXT:       ContainingCsectSymbolIndex: 8
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_LD (0x2)
; XCOFF32-NEXT:       StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+16]]
; XCOFF32-NEXT:     Name: .foo_weak_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x30
; XCOFF32-NEXT:     Section: .text
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+17]]
; XCOFF32-NEXT:       ContainingCsectSymbolIndex: 8
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_LD (0x2)
; XCOFF32-NEXT:       StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+18]]
; XCOFF32-NEXT:     Name: .main
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x40
; XCOFF32-NEXT:     Section: .text
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+19]]
; XCOFF32-NEXT:       ContainingCsectSymbolIndex: 8
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_LD (0x2)
; XCOFF32-NEXT:       StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+20]]
; XCOFF32-NEXT:     Name: .data
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x98
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+21]]
; XCOFF32-NEXT:       SectionLen: 12
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_RW (0x5)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+22]]
; XCOFF32-NEXT:     Name: b
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x98
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+23]]
; XCOFF32-NEXT:       ContainingCsectSymbolIndex: 20
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_LD (0x2)
; XCOFF32-NEXT:       StorageMappingClass: XMC_RW (0x5)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+24]]
; XCOFF32-NEXT:     Name: b_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x9C
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+25]]
; XCOFF32-NEXT:       ContainingCsectSymbolIndex: 20
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_LD (0x2)
; XCOFF32-NEXT:       StorageMappingClass: XMC_RW (0x5)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+26]]
; XCOFF32-NEXT:     Name: foo_p
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xA0
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+27]]
; XCOFF32-NEXT:       ContainingCsectSymbolIndex: 20
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_LD (0x2)
; XCOFF32-NEXT:       StorageMappingClass: XMC_RW (0x5)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+28]]
; XCOFF32-NEXT:     Name: foo
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xA4
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+29]]
; XCOFF32-NEXT:       SectionLen: 12
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_DS (0xA)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+30]]
; XCOFF32-NEXT:     Name: foo_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xB0
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+31]]
; XCOFF32-NEXT:       SectionLen: 12
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_DS (0xA)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+32]]
; XCOFF32-NEXT:     Name: foo_protected
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xBC
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x3000
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+33]]
; XCOFF32-NEXT:       SectionLen: 12
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_DS (0xA)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+34]]
; XCOFF32-NEXT:     Name: foo_weak_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xC8
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x2000
; XCOFF32-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+35]]
; XCOFF32-NEXT:       SectionLen: 12
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_DS (0xA)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+36]]
; XCOFF32-NEXT:     Name: main
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xD4
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+37]]
; XCOFF32-NEXT:       SectionLen: 12
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_DS (0xA)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+38]]
; XCOFF32-NEXT:     Name: TOC
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xE0
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+39]]
; XCOFF32-NEXT:       SectionLen: 0
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_TC0 (0xF)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+40]]
; XCOFF32-NEXT:     Name: b_h
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xE0
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+41]]
; XCOFF32-NEXT:       SectionLen: 4
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_TC (0x3)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT:   Symbol {
; XCOFF32-NEXT:     Index: [[#Index+42]]
; XCOFF32-NEXT:     Name: foo_p
; XCOFF32-NEXT:     Value (RelocatableAddress): 0xE4
; XCOFF32-NEXT:     Section: .data
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+43]]
; XCOFF32-NEXT:       SectionLen: 4
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 2
; XCOFF32-NEXT:       SymbolType: XTY_SD (0x1)
; XCOFF32-NEXT:       StorageMappingClass: XMC_TC (0x3)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
; XCOFF32-NEXT:   }
; XCOFF32-NEXT: ]
