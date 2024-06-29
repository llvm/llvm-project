; Testing 32-bit and 64-bit exception section entries, no exception auxilliary
; entries should be produced as no debug information is specified.
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff -filetype=obj -o %t_32.o < %s
; RUN: llvm-readobj --exception-section %t_32.o | FileCheck %s --check-prefix=EXCEPT
; RUN: llvm-readobj --section-headers %t_32.o | FileCheck %s --check-prefix=READ
; RUN: llvm-readobj --syms %t_32.o | FileCheck %s --check-prefix=SYMS

; RUN: llc -mtriple=powerpc64-unknown-aix -filetype=obj -o %t_64.o < %s
; RUN: llvm-readobj --exception-section %t_64.o | FileCheck %s --check-prefix=EXCEPT64
; RUN: llvm-readobj --section-headers %t_64.o | FileCheck %s --check-prefix=READ64
; RUN: llvm-readobj --syms %t_64.o | FileCheck %s --check-prefix=SYMS64

!1 = !{!"ppc-trap-reason", !"1", !"2"}
declare void @llvm.ppc.trap(i32 %a)
define dso_local void @sub_test() {
  call void @llvm.ppc.trap(i32 1), !annotation !1
  ret void
}
define dso_local void @test__trap_annotation(i32 %a) {
  call void @llvm.ppc.trap(i32 %a), !annotation !1
  call void @sub_test()
  call void @llvm.ppc.trap(i32 %a), !annotation !1
  ret void
}

; EXCEPT:       Exception section {
; EXCEPT-NEXT:    Symbol: .sub_test
; EXCEPT-NEXT:    LangID: 0
; EXCEPT-NEXT:    Reason: 0
; EXCEPT-NEXT:    Trap Instr Addr: 0x4
; EXCEPT-NEXT:    LangID: 1
; EXCEPT-NEXT:    Reason: 2
; EXCEPT-NEXT:    Symbol: .test__trap_annotation
; EXCEPT-NEXT:    LangID: 0
; EXCEPT-NEXT:    Reason: 0
; EXCEPT-NEXT:    Trap Instr Addr: 0x3C
; EXCEPT-NEXT:    LangID: 1
; EXCEPT-NEXT:    Reason: 2
; EXCEPT-NEXT:    Trap Instr Addr: 0x44
; EXCEPT-NEXT:    LangID: 1
; EXCEPT-NEXT:    Reason: 2
; EXCEPT-NEXT:  }

; There are multiple "Section {" lines in the readobj output so we need to start this READ check
; on a unique line (Type: STYP_DATA (0x40)) so that the checks know where to start reading
; READ:           Type: STYP_DATA (0x40)
; READ-NEXT:    }
; READ-NEXT:    Section {
; READ-NEXT:      Index: 3
; READ-NEXT:      Name: .except
; READ-NEXT:      PhysicalAddress: 0x0
; READ-NEXT:      VirtualAddress: 0x0
; READ-NEXT:      Size: 0x1E
; READ-NEXT:      RawDataOffset: 0x12C
; READ-NEXT:      RelocationPointer: 0x0
; READ-NEXT:      LineNumberPointer: 0x0
; READ-NEXT:      NumberOfRelocations: 0
; READ-NEXT:      NumberOfLineNumbers: 0
; READ-NEXT:      Type: STYP_EXCEPT (0x100)
; READ-NEXT:    }
; READ-NEXT:  ]

; SYMS:           Index: [[#IND:]]{{.*}}{{[[:space:]] *}}Name: .sub_test
; SYMS-NEXT:      Value (RelocatableAddress): 0x0
; SYMS-NEXT:      Section: .text
; SYMS-NEXT:      Type: 0x20
; SYMS-NEXT:      StorageClass: C_EXT (0x2)
; SYMS-NEXT:      NumberOfAuxEntries: 2
; SYMS-NEXT:      Function Auxiliary Entry {
; SYMS-NEXT:        Index: [[#IND+1]]
; SYMS-NEXT:        OffsetToExceptionTable: 0x12C
; SYMS-NEXT:        SizeOfFunction: 0xC
; SYMS-NEXT:        PointerToLineNum: 0x0
; SYMS-NEXT:        SymbolIndexOfNextBeyond: [[#IND+3]]
; SYMS-NEXT:      }
; SYMS-NEXT:      CSECT Auxiliary Entry {
; SYMS-NEXT:        Index: [[#IND+2]]
; SYMS-NEXT:        ContainingCsectSymbolIndex: [[#IND-2]]
; SYMS-NEXT:        ParameterHashIndex: 0x0
; SYMS-NEXT:        TypeChkSectNum: 0x0
; SYMS-NEXT:        SymbolAlignmentLog2: 0
; SYMS-NEXT:        SymbolType: XTY_LD (0x2)
; SYMS-NEXT:        StorageMappingClass: XMC_PR (0x0)
; SYMS-NEXT:        StabInfoIndex: 0x0
; SYMS-NEXT:        StabSectNum: 0x0
; SYMS-NEXT:      }
; SYMS-NEXT:    }
; SYMS-NEXT:    Symbol {
; SYMS-NEXT:      Index: [[#IND+3]]
; SYMS-NEXT:      Name: .test__trap_annotation
; SYMS-NEXT:      Value (RelocatableAddress): 0x28
; SYMS-NEXT:      Section: .text
; SYMS-NEXT:      Type: 0x20
; SYMS-NEXT:      StorageClass: C_EXT (0x2)
; SYMS-NEXT:      NumberOfAuxEntries: 2
; SYMS-NEXT:      Function Auxiliary Entry {
; SYMS-NEXT:        Index: [[#IND+4]]
; SYMS-NEXT:        OffsetToExceptionTable: 0x138
; SYMS-NEXT:        SizeOfFunction: 0x34
; SYMS-NEXT:        PointerToLineNum: 0x0
; SYMS-NEXT:        SymbolIndexOfNextBeyond: [[#IND+6]]
; SYMS-NEXT:      }
; SYMS-NEXT:      CSECT Auxiliary Entry {
; SYMS-NEXT:        Index: [[#IND+5]]
; SYMS-NEXT:        ContainingCsectSymbolIndex: [[#IND-2]]
; SYMS-NEXT:        ParameterHashIndex: 0x0
; SYMS-NEXT:        TypeChkSectNum: 0x0
; SYMS-NEXT:        SymbolAlignmentLog2: 0
; SYMS-NEXT:        SymbolType: XTY_LD (0x2)
; SYMS-NEXT:        StorageMappingClass: XMC_PR (0x0)
; SYMS-NEXT:        StabInfoIndex: 0x0
; SYMS-NEXT:        StabSectNum: 0x0
; SYMS-NEXT:      }
; SYMS-NEXT:    }

; EXCEPT64:       Exception section {
; EXCEPT64-NEXT:    Symbol: .sub_test
; EXCEPT64-NEXT:    LangID: 0
; EXCEPT64-NEXT:    Reason: 0
; EXCEPT64-NEXT:    Trap Instr Addr: 0x4
; EXCEPT64-NEXT:    LangID: 1
; EXCEPT64-NEXT:    Reason: 2
; EXCEPT64-NEXT:    Symbol: .test__trap_annotation
; EXCEPT64-NEXT:    LangID: 0
; EXCEPT64-NEXT:    Reason: 0
; EXCEPT64-NEXT:    Trap Instr Addr: 0x3C
; EXCEPT64-NEXT:    LangID: 1
; EXCEPT64-NEXT:    Reason: 2
; EXCEPT64-NEXT:    Trap Instr Addr: 0x44
; EXCEPT64-NEXT:    LangID: 1
; EXCEPT64-NEXT:    Reason: 2
; EXCEPT64-NEXT:  }

; READ64:           Type: STYP_DATA (0x40)
; READ64-NEXT:    }
; READ64-NEXT:    Section {
; READ64-NEXT:      Index: 3
; READ64-NEXT:      Name: .except
; READ64-NEXT:      PhysicalAddress: 0x0
; READ64-NEXT:      VirtualAddress: 0x0
; READ64-NEXT:      Size: 0x32
; READ64-NEXT:      RawDataOffset: 0x1A8
; READ64-NEXT:      RelocationPointer: 0x0
; READ64-NEXT:      LineNumberPointer: 0x0
; READ64-NEXT:      NumberOfRelocations: 0
; READ64-NEXT:      NumberOfLineNumbers: 0
; READ64-NEXT:      Type: STYP_EXCEPT (0x100)
; READ64-NEXT:    }
; READ64-NEXT:  ]

; SYMS64:           Index: [[#IND:]]{{.*}}{{[[:space:]] *}}Name: .sub_test
; SYMS64-NEXT:      Value (RelocatableAddress): 0x0
; SYMS64-NEXT:      Section: .text
; SYMS64-NEXT:      Type: 0x0
; SYMS64-NEXT:      StorageClass: C_EXT (0x2)
; SYMS64-NEXT:      NumberOfAuxEntries: 2
; SYMS64-NEXT:      Function Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+1]]
; SYMS64-NEXT:        SizeOfFunction: 0x18
; SYMS64-NEXT:        PointerToLineNum: 0x0
; SYMS64-NEXT:        SymbolIndexOfNextBeyond: [[#IND+3]]
; SYMS64-NEXT:        Auxiliary Type: AUX_FCN (0xFE)
; SYMS64-NEXT:      }
; SYMS64-NEXT:      CSECT Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+2]]
; SYMS64-NEXT:        ContainingCsectSymbolIndex: [[#IND-2]]
; SYMS64-NEXT:        ParameterHashIndex: 0x0
; SYMS64-NEXT:        TypeChkSectNum: 0x0
; SYMS64-NEXT:        SymbolAlignmentLog2: 0
; SYMS64-NEXT:        SymbolType: XTY_LD (0x2)
; SYMS64-NEXT:        StorageMappingClass: XMC_PR (0x0)
; SYMS64-NEXT:        Auxiliary Type: AUX_CSECT (0xFB)
; SYMS64-NEXT:      }
; SYMS64-NEXT:    }
; SYMS64-NEXT:    Symbol {
; SYMS64-NEXT:      Index: [[#IND+3]]
; SYMS64-NEXT:      Name: .test__trap_annotation
; SYMS64-NEXT:      Value (RelocatableAddress): 0x28
; SYMS64-NEXT:      Section: .text
; SYMS64-NEXT:      Type: 0x0
; SYMS64-NEXT:      StorageClass: C_EXT (0x2)
; SYMS64-NEXT:      NumberOfAuxEntries: 2
; SYMS64-NEXT:      Function Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+4]]
; SYMS64-NEXT:        SizeOfFunction: 0x68
; SYMS64-NEXT:        PointerToLineNum: 0x0
; SYMS64-NEXT:        SymbolIndexOfNextBeyond: [[#IND+6]]
; SYMS64-NEXT:        Auxiliary Type: AUX_FCN (0xFE)
; SYMS64-NEXT:      }
; SYMS64-NEXT:      CSECT Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+5]]
; SYMS64-NEXT:        ContainingCsectSymbolIndex: [[#IND-2]]
; SYMS64-NEXT:        ParameterHashIndex: 0x0
; SYMS64-NEXT:        TypeChkSectNum: 0x0
; SYMS64-NEXT:        SymbolAlignmentLog2: 0
; SYMS64-NEXT:        SymbolType: XTY_LD (0x2)
; SYMS64-NEXT:        StorageMappingClass: XMC_PR (0x0)
; SYMS64-NEXT:        Auxiliary Type: AUX_CSECT (0xFB)
; SYMS64-NEXT:      }
; SYMS64-NEXT:    }
