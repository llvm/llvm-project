; This file tests the codegen of tls variables in AIX XCOFF object files

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers %t.o | FileCheck --check-prefix=SECTION %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefixes=SYMS,SYMS-DATASECT %s
; RUN: llvm-objdump -D --symbol-description %t.o | FileCheck --check-prefixes=OBJDUMP-DATASECT %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers %t.o | FileCheck --check-prefix=SECTION %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefixes=SYMS,SYMS-NODATASECT %s
; RUN: llvm-objdump -D --symbol-description %t.o | FileCheck --check-prefixes=OBJDUMP-NODATASECT %s

;; FIXME: currently only fileHeader and sectionHeaders are supported in XCOFF64.

; SECTION:      File: {{.*}}aix-tls-xcoff-variables.ll.tmp.o
; SECTION-NEXT: Format: aixcoff-rs6000
; SECTION-NEXT: Arch: powerpc
; SECTION-NEXT: AddressSize: 32bit
; SECTION-NEXT: Sections [
; SECTION-NEXT:   Section {
; SECTION-NEXT:     Index: 1
; SECTION-NEXT:     Name: .text
; SECTION-NEXT:     PhysicalAddress: 0x0
; SECTION-NEXT:     VirtualAddress: 0x0
; SECTION-NEXT:     Size: 0x4
; SECTION-NEXT:     RawDataOffset: 0x8C
; SECTION-NEXT:     RelocationPointer: 0x0
; SECTION-NEXT:     LineNumberPointer: 0x0
; SECTION-NEXT:     NumberOfRelocations: 0
; SECTION-NEXT:     NumberOfLineNumbers: 0
; SECTION-NEXT:     Type: STYP_TEXT (0x20)
; SECTION-NEXT:   }
; SECTION-NEXT:   Section {
; SECTION-NEXT:     Index: 2
; SECTION-NEXT:     Name: .tdata
; SECTION-NEXT:     PhysicalAddress: 0x0
; SECTION-NEXT:     VirtualAddress: 0x0
; SECTION-NEXT:     Size: 0x30
; SECTION-NEXT:     RawDataOffset: 0x90
; SECTION-NEXT:     RelocationPointer: 0x0
; SECTION-NEXT:     LineNumberPointer: 0x0
; SECTION-NEXT:     NumberOfRelocations: 0
; SECTION-NEXT:     NumberOfLineNumbers: 0
; SECTION-NEXT:     Type: STYP_TDATA (0x400)
; SECTION-NEXT:   }
; SECTION-NEXT:   Section {
; SECTION-NEXT:     Index: 3
; SECTION-NEXT:     Name: .tbss
; SECTION-NEXT:     PhysicalAddress: 0x30
; SECTION-NEXT:     VirtualAddress: 0x30
; SECTION-NEXT:     Size: 0x18
; SECTION-NEXT:     RawDataOffset: 0x0
; SECTION-NEXT:     RelocationPointer: 0x0
; SECTION-NEXT:     LineNumberPointer: 0x0
; SECTION-NEXT:     NumberOfRelocations: 0
; SECTION-NEXT:     NumberOfLineNumbers: 0
; SECTION-NEXT:     Type: STYP_TBSS (0x800)
; SECTION-NEXT:   }
; SECTION-NEXT: ]


; SYMS:      File: {{.*}}aix-tls-xcoff-variables.ll.tmp.o
; SYMS-NEXT: Format: aixcoff-rs6000
; SYMS-NEXT: Arch: powerpc
; SYMS-NEXT: AddressSize: 32bit
; SYMS-NEXT: Symbols [
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: 0
; SYMS-NEXT:     Name: .file
; SYMS-NEXT:     Value (SymbolTableIndex): 0x0
; SYMS-NEXT:     Section: N_DEBUG
; SYMS-NEXT:     Source Language ID: TB_C (0x0)
; SYMS-NEXT:     CPU Version ID: 0x0
; SYMS-NEXT:     StorageClass: C_FILE (0x67)
; SYMS-NEXT:     NumberOfAuxEntries: 0
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#INDX:]]
; SYMS-NEXT:     Name: tls_global_int_external_uninitialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: N_UNDEF
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+1]]
; SYMS-NEXT:       SectionLen: 0
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_ER (0x0)
; SYMS-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }
; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#INDX+2]]
; SYMS-NEXT:     Name: tls_global_double_external_uninitialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: N_UNDEF
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+3]]
; SYMS-NEXT:       SectionLen: 0
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_ER (0x0)
; SYMS-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS-NEXT:   Symbol {
; SYMS-NEXT:     Index: [[#INDX+4]]
; SYMS-NEXT:     Name: .text
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-NEXT:       Index: [[#INDX+5]]
; SYMS-NEXT:       SectionLen: 0
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NEXT:       SymbolType: XTY_SD (0x1)
; SYMS-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS-NODATASECT:        Symbol {
; SYMS-NODATASECT-NEXT:     Index: [[#INDX+6]]
; SYMS-NODATASECT-NEXT:     Name: .rodata
; SYMS-NODATASECT-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NODATASECT-NEXT:     Section: .text
; SYMS-NODATASECT-NEXT:     Type: 0x0
; SYMS-NODATASECT-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NODATASECT-NEXT:     NumberOfAuxEntries: 1
; SYMS-NODATASECT-NEXT:     CSECT Auxiliary Entry {
; SYMS-NODATASECT-NEXT:       Index: [[#INDX+7]]
; SYMS-NODATASECT-NEXT:       SectionLen: 4
; SYMS-NODATASECT-NEXT:       ParameterHashIndex: 0x0
; SYMS-NODATASECT-NEXT:       TypeChkSectNum: 0x0
; SYMS-NODATASECT-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NODATASECT-NEXT:       SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT-NEXT:       StorageMappingClass: XMC_RO (0x1)
; SYMS-NODATASECT-NEXT:       StabInfoIndex: 0x0
; SYMS-NODATASECT-NEXT:       StabSectNum: 0x0
; SYMS-NODATASECT-NEXT:     }
; SYMS-NODATASECT-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+6]]
; SYMS-NODATASECT: Index: [[#INDX+8]]
; SYMS-NEXT:     Name: const_ivar
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .text
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+7]]
; SYMS-DATASECT-NEXT:   SectionLen: 4
; SYMS-NODATASECT:      Index: [[#INDX+9]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+6]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 2
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_RO (0x1)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS-NODATASECT:        Symbol {
; SYMS-NODATASECT-NEXT:     Index: [[#INDX+10]]
; SYMS-NODATASECT-NEXT:     Name: .tdata
; SYMS-NODATASECT-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NODATASECT-NEXT:     Section: .tdata
; SYMS-NODATASECT-NEXT:     Type: 0x0
; SYMS-NODATASECT-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NODATASECT-NEXT:     NumberOfAuxEntries: 1
; SYMS-NODATASECT-NEXT:     CSECT Auxiliary Entry {
; SYMS-NODATASECT-NEXT:       Index: [[#INDX+11]]
; SYMS-NODATASECT-NEXT:       SectionLen: 48
; SYMS-NODATASECT-NEXT:       ParameterHashIndex: 0x0
; SYMS-NODATASECT-NEXT:       TypeChkSectNum: 0x0
; SYMS-NODATASECT-NEXT:       SymbolAlignmentLog2: 3
; SYMS-NODATASECT-NEXT:       SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYMS-NODATASECT-NEXT:       StabInfoIndex: 0x0
; SYMS-NODATASECT-NEXT:       StabSectNum: 0x0
; SYMS-NODATASECT-NEXT:     }
; SYMS-NODATASECT-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+8]]
; SYMS-NODATASECT: Index: [[#INDX+12]]
; SYMS:          Name: tls_global_int_external_val_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+9]]
; SYMS-DATASECT-NEXT:   SectionLen: 4
; SYMS-NODATASECT:      Index: [[#INDX+13]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 2
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+10]]
; SYMS-NODATASECT: Index: [[#INDX+14]]
; SYMS:          Name: tls_global_alias_int_external_val_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x0
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+11]]
; SYMS-DATASECT-NEXT:   ContainingCsectSymbolIndex: [[#INDX+8]]
; SYMS-NODATASECT:      Index: [[#INDX+15]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 0
; SYMS-NEXT:       SymbolType: XTY_LD (0x2)
; SYMS-NEXT:       StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+12]]
; SYMS-NODATASECT: Index: [[#INDX+16]]
; SYMS:          Name: tls_global_int_external_zero_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x4
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+13]]
; SYMS-DATASECT-NEXT:   SectionLen: 4
; SYMS-NODATASECT:      Index: [[#INDX+17]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 2
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+14]]
; SYMS-NODATASECT: Index: [[#INDX+18]]
; SYMS:          Name: tls_global_int_local_val_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x8
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+15]]
; SYMS-DATASECT-NEXT:   SectionLen: 4
; SYMS-NODATASECT:      Index: [[#INDX+19]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 2
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+16]]
; SYMS-NODATASECT: Index: [[#INDX+20]]
; SYMS:          Name: tls_global_int_weak_zero_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0xC
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+17]]
; SYMS-DATASECT-NEXT:   SectionLen: 4
; SYMS-NODATASECT:      Index: [[#INDX+21]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 2
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+18]]
; SYMS-NODATASECT: Index: [[#INDX+22]]
; SYMS:          Name: tls_global_int_weak_val_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x10
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+19]]
; SYMS-DATASECT-NEXT:   SectionLen: 4
; SYMS-NODATASECT:      Index: [[#INDX+23]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 2
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+20]]
; SYMS-NODATASECT: Index: [[#INDX+24]]
; SYMS:          Name: tls_global_long_long_internal_val_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x18
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+21]]
; SYMS-DATASECT-NEXT:   SectionLen: 8
; SYMS-NODATASECT:      Index: [[#INDX+25]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 3
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+22]]
; SYMS-NODATASECT: Index: [[#INDX+26]]
; SYMS:          Name: tls_global_long_long_weak_val_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x20
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+23]]
; SYMS-DATASECT-NEXT:   SectionLen: 8
; SYMS-NODATASECT:      Index: [[#INDX+27]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 3
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+24]]
; SYMS-NODATASECT: Index: [[#INDX+28]]
; SYMS:          Name: tls_global_long_long_weak_zero_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x28
; SYMS-NEXT:     Section: .tdata
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_WEAKEXT (0x6F)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:        Index: [[#INDX+25]]
; SYMS-DATASECT-NEXT:   SectionLen: 8
; SYMS-NODATASECT:      Index: [[#INDX+29]]
; SYMS-NODATASECT-NEXT: ContainingCsectSymbolIndex: [[#INDX+10]]
; SYMS:            ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-DATASECT:        SymbolAlignmentLog2: 3
; SYMS-DATASECT-NEXT:   SymbolType: XTY_SD (0x1)
; SYMS-NODATASECT:      SymbolAlignmentLog2: 0
; SYMS-NODATASECT-NEXT: SymbolType: XTY_LD (0x2)
; SYMS:            StorageMappingClass: XMC_TL (0x14)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+26]]
; SYMS-NODATASECT: Index: [[#INDX+30]]
; SYMS:          Name: tls_global_int_local_zero_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x30
; SYMS-NEXT:     Section: .tbss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:   Index: [[#INDX+27]]
; SYMS-NODATASECT: Index: [[#INDX+31]]
; SYMS:            SectionLen: 4
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+28]]
; SYMS-NODATASECT: Index: [[#INDX+32]]
; SYMS:          Name: tls_global_int_common_zero_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x34
; SYMS-NEXT:     Section: .tbss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:   Index: [[#INDX+29]]
; SYMS-NODATASECT: Index: [[#INDX+33]]
; SYMS:            SectionLen: 4
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 2
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+30]]
; SYMS-NODATASECT: Index: [[#INDX+34]]
; SYMS:          Name: tls_global_double_common_zero_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x38
; SYMS-NEXT:     Section: .tbss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_EXT (0x2)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:   Index: [[#INDX+31]]
; SYMS-NODATASECT: Index: [[#INDX+35]]
; SYMS:            SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 3
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }

; SYMS:        Symbol {
; SYMS-DATASECT:   Index: [[#INDX+32]]
; SYMS-NODATASECT: Index: [[#INDX+36]]
; SYMS:          Name: tls_global_long_long_internal_zero_initialized
; SYMS-NEXT:     Value (RelocatableAddress): 0x40
; SYMS-NEXT:     Section: .tbss
; SYMS-NEXT:     Type: 0x0
; SYMS-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:     NumberOfAuxEntries: 1
; SYMS-NEXT:     CSECT Auxiliary Entry {
; SYMS-DATASECT:   Index: [[#INDX+33]]
; SYMS-NODATASECT: Index: [[#INDX+37]]
; SYMS:            SectionLen: 8
; SYMS-NEXT:       ParameterHashIndex: 0x0
; SYMS-NEXT:       TypeChkSectNum: 0x0
; SYMS-NEXT:       SymbolAlignmentLog2: 3
; SYMS-NEXT:       SymbolType: XTY_CM (0x3)
; SYMS-NEXT:       StorageMappingClass: XMC_UL (0x15)
; SYMS-NEXT:       StabInfoIndex: 0x0
; SYMS-NEXT:       StabSectNum: 0x0
; SYMS-NEXT:     }
; SYMS-NEXT:   }


; OBJDUMP-DATASECT:        Disassembly of section .text:
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000000 (idx: 7) const_ivar[RO]:
; OBJDUMP-DATASECT-NEXT:   0: 00 00 00 06   <unknown>
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   Disassembly of section .tdata:
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000000 (idx: 11) tls_global_alias_int_external_val_initialized:
; OBJDUMP-DATASECT-NEXT:   0: 00 00 00 01   <unknown>
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000004 (idx: 13) tls_global_int_external_zero_initialized[TL]:
; OBJDUMP-DATASECT-NEXT:   4: 00 00 00 00   <unknown>
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000008 (idx: 15) tls_global_int_local_val_initialized[TL]:
; OBJDUMP-DATASECT-NEXT:   8: 00 00 00 02   <unknown>
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   0000000c (idx: 17) tls_global_int_weak_zero_initialized[TL]:
; OBJDUMP-DATASECT-NEXT:   c: 00 00 00 00   <unknown>
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000010 (idx: 19) tls_global_int_weak_val_initialized[TL]:
; OBJDUMP-DATASECT-NEXT:   10: 00 00 00 01   <unknown>
; OBJDUMP-DATASECT-NEXT:   14: 00 00 00 00   <unknown>
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000018 (idx: 21) tls_global_long_long_internal_val_initialized[TL]:
; OBJDUMP-DATASECT-NEXT:   18: 00 00 00 00   <unknown>
; OBJDUMP-DATASECT-NEXT:   1c: 00 00 00 01   <unknown>
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000020 (idx: 23) tls_global_long_long_weak_val_initialized[TL]:
; OBJDUMP-DATASECT-NEXT:   20: 00 00 00 00   <unknown>
; OBJDUMP-DATASECT-NEXT:   24: 00 00 00 01   <unknown>
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000028 (idx: 25) tls_global_long_long_weak_zero_initialized[TL]:
; OBJDUMP-DATASECT-NEXT:   ...
; OBJDUMP-DATASECT-EMPTY:

; OBJDUMP-DATASECT-NEXT:   Disassembly of section .tbss:
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000030 (idx: 27) tls_global_int_local_zero_initialized[UL]:
; OBJDUMP-DATASECT-NEXT:   ...
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000034 (idx: 29) tls_global_int_common_zero_initialized[UL]:
; OBJDUMP-DATASECT-NEXT:   ...
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000038 (idx: 31) tls_global_double_common_zero_initialized[UL]
; OBJDUMP-DATASECT-NEXT:   ...
; OBJDUMP-DATASECT-EMPTY:
; OBJDUMP-DATASECT-NEXT:   00000040 (idx: 33) tls_global_long_long_internal_zero_initialized[UL]:
; OBJDUMP-DATASECT-NEXT:   ...

; OBJDUMP-NODATASECT:       Disassembly of section .text:
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000000 (idx: 9) const_ivar:
; OBJDUMP-NODATASECT-NEXT:  0: 00 00 00 06   <unknown>
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  Disassembly of section .tdata:
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT:       00000000 (idx: 13) tls_global_int_external_val_initialized:
; OBJDUMP-NODATASECT-NEXT:  0: 00 00 00 01   <unknown>
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000004 (idx: 17) tls_global_int_external_zero_initialized:
; OBJDUMP-NODATASECT-NEXT:  4: 00 00 00 00   <unknown>
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000008 (idx: 19) tls_global_int_local_val_initialized:
; OBJDUMP-NODATASECT-NEXT:  8: 00 00 00 02   <unknown>
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  0000000c (idx: 21) tls_global_int_weak_zero_initialized:
; OBJDUMP-NODATASECT-NEXT:  c: 00 00 00 00   <unknown>
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000010 (idx: 23) tls_global_int_weak_val_initialized:
; OBJDUMP-NODATASECT-NEXT:  10: 00 00 00 01   <unknown>
; OBJDUMP-NODATASECT-NEXT:  14: 00 00 00 00   <unknown>
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000018 (idx: 25) tls_global_long_long_internal_val_initialized:
; OBJDUMP-NODATASECT-NEXT:  18: 00 00 00 00   <unknown>
; OBJDUMP-NODATASECT-NEXT:  1c: 00 00 00 01   <unknown>
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000020 (idx: 27) tls_global_long_long_weak_val_initialized:
; OBJDUMP-NODATASECT-NEXT:  20: 00 00 00 00   <unknown>
; OBJDUMP-NODATASECT-NEXT:  24: 00 00 00 01   <unknown>
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000028 (idx: 29) tls_global_long_long_weak_zero_initialized:
; OBJDUMP-NODATASECT-NEXT:  ...
; OBJDUMP-NODATASECT-EMPTY:

; OBJDUMP-NODATASECT-NEXT:  Disassembly of section .tbss:
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000030 (idx: 31) tls_global_int_local_zero_initialized[UL]:
; OBJDUMP-NODATASECT-NEXT:  ...
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000034 (idx: 33) tls_global_int_common_zero_initialized[UL]:
; OBJDUMP-NODATASECT-NEXT:  ...
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000038 (idx: 35) tls_global_double_common_zero_initialized[UL]:
; OBJDUMP-NODATASECT-NEXT:  ...
; OBJDUMP-NODATASECT-EMPTY:
; OBJDUMP-NODATASECT-NEXT:  00000040 (idx: 37) tls_global_long_long_internal_zero_initialized[UL]:
; OBJDUMP-NODATASECT-NEXT:  ...

@tls_global_int_external_val_initialized = thread_local global i32 1, align 4
@tls_global_int_external_zero_initialized = thread_local global i32 0, align 4
@tls_global_int_local_val_initialized = internal thread_local global i32 2, align 4
@tls_global_int_local_zero_initialized = internal thread_local global i32 0, align 4
@tls_global_int_weak_zero_initialized = weak thread_local global i32 0, align 4
@tls_global_int_common_zero_initialized = common thread_local global i32 0, align 4
@tls_global_int_weak_val_initialized = weak thread_local global i32 1, align 4
@tls_global_int_external_uninitialized = external thread_local global i32, align 4
@tls_global_double_common_zero_initialized = common thread_local global double 0.000000e+00, align 8
@tls_global_double_external_uninitialized = external thread_local global i64, align 8
@tls_global_long_long_internal_val_initialized = internal thread_local global i64 1, align 8
@tls_global_long_long_internal_zero_initialized = internal thread_local global i64 0, align 8
@tls_global_long_long_weak_val_initialized = weak thread_local global i64 1, align 8
@tls_global_long_long_weak_zero_initialized = weak thread_local global i64 0, align 8
@tls_global_alias_int_external_val_initialized = thread_local alias i32, i32* @tls_global_int_external_val_initialized
@const_ivar = constant i32 6, align 4
