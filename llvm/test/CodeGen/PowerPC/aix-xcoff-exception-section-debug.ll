; This file contains exception section testing for when debug information is present.
; The 32-bit test should not print exception auxilliary entries because they are a 64-bit only feature.
; Exception auxilliary entries are present in the 64-bit tests because 64-bit && debug enabled are the requirements.
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff -filetype=obj -o %t_32.o < %s
; RUN: llvm-readobj --syms %t_32.o | FileCheck %s --check-prefix=SYMS32
; RUN: llc -mtriple=powerpc64-unknown-aix -filetype=obj -o %t_32.o < %s
; RUN: llvm-readobj --syms %t_32.o | FileCheck %s --check-prefix=SYMS64

; If any debug information is included in a module and is XCOFF64, exception auxilliary entries are emitted

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 3}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "ASTI IR translator", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!3 = !DIFile(filename: "t.f", directory: ".")
!4 = distinct !DISubprogram(name: "test__trap_annotation_debug", linkageName: "test__trap_annotation_debug", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 4, column: 1, scope: !4)
!8 = !{!"ppc-trap-reason", !"1", !"2"}
declare void @llvm.ppc.trap(i32 %a)
define dso_local void @sub_test() {
  call void @llvm.ppc.trap(i32 1), !annotation !8
  ret void
}
define dso_local void @test__trap_annotation_debug(i32 %a) !dbg !4 {
  call void @llvm.ppc.trap(i32 %a), !annotation !8
  call void @sub_test()
  call void @llvm.ppc.trap(i32 %a), !annotation !8
  ret void
}

; SYMS32:           Index: [[#IND:]]{{.*}}{{[[:space:]] *}}Name: .sub_test
; SYMS32-NEXT:      Value (RelocatableAddress): 0x0
; SYMS32-NEXT:      Section: .text
; SYMS32-NEXT:      Type: 0x20
; SYMS32-NEXT:      StorageClass: C_EXT (0x2)
; SYMS32-NEXT:      NumberOfAuxEntries: 2
; SYMS32-NEXT:      Function Auxiliary Entry {
; SYMS32-NEXT:        Index: [[#IND+1]]
; SYMS32-NEXT:        OffsetToExceptionTable: 0x2A8
; SYMS32-NEXT:        SizeOfFunction: 0xC
; SYMS32-NEXT:        PointerToLineNum: 0x0
; SYMS32-NEXT:        SymbolIndexOfNextBeyond: [[#IND+3]] 
; SYMS32-NEXT:      }
; SYMS32-NEXT:      CSECT Auxiliary Entry {
; SYMS32-NEXT:        Index: [[#IND+2]]
; SYMS32-NEXT:        ContainingCsectSymbolIndex: [[#IND-2]]
; SYMS32-NEXT:        ParameterHashIndex: 0x0
; SYMS32-NEXT:        TypeChkSectNum: 0x0
; SYMS32-NEXT:        SymbolAlignmentLog2: 0
; SYMS32-NEXT:        SymbolType: XTY_LD (0x2)
; SYMS32-NEXT:        StorageMappingClass: XMC_PR (0x0)
; SYMS32-NEXT:        StabInfoIndex: 0x0
; SYMS32-NEXT:        StabSectNum: 0x0
; SYMS32-NEXT:      }
; SYMS32-NEXT:    }
; SYMS32-NEXT:    Symbol {
; SYMS32-NEXT:      Index: [[#IND+3]]
; SYMS32-NEXT:      Name: .test__trap_annotation
; SYMS32-NEXT:      Value (RelocatableAddress): 0x28
; SYMS32-NEXT:      Section: .text
; SYMS32-NEXT:      Type: 0x20
; SYMS32-NEXT:      StorageClass: C_EXT (0x2)
; SYMS32-NEXT:      NumberOfAuxEntries: 2
; SYMS32-NEXT:      Function Auxiliary Entry {
; SYMS32-NEXT:        Index: [[#IND+4]]
; SYMS32-NEXT:        OffsetToExceptionTable: 0x2B4
; SYMS32-NEXT:        SizeOfFunction: 0x34
; SYMS32-NEXT:        PointerToLineNum: 0x0
; SYMS32-NEXT:        SymbolIndexOfNextBeyond: [[#IND+6]]
; SYMS32-NEXT:      }
; SYMS32-NEXT:      CSECT Auxiliary Entry {
; SYMS32-NEXT:        Index: [[#IND+5]]
; SYMS32-NEXT:        ContainingCsectSymbolIndex: [[#IND-2]]
; SYMS32-NEXT:        ParameterHashIndex: 0x0
; SYMS32-NEXT:        TypeChkSectNum: 0x0
; SYMS32-NEXT:        SymbolAlignmentLog2: 0
; SYMS32-NEXT:        SymbolType: XTY_LD (0x2)
; SYMS32-NEXT:        StorageMappingClass: XMC_PR (0x0)
; SYMS32-NEXT:        StabInfoIndex: 0x0
; SYMS32-NEXT:        StabSectNum: 0x0
; SYMS32-NEXT:      }
; SYMS32-NEXT:    }

; SYMS64:           Index: [[#IND:]]{{.*}}{{[[:space:]] *}}Name: .sub_test
; SYMS64-NEXT:      Value (RelocatableAddress): 0x0
; SYMS64-NEXT:      Section: .text
; SYMS64-NEXT:      Type: 0x0
; SYMS64-NEXT:      StorageClass: C_EXT (0x2)
; SYMS64-NEXT:      NumberOfAuxEntries: 3
; SYMS64-NEXT:      Exception Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+1]]
; SYMS64-NEXT:        OffsetToExceptionTable: 0x398
; SYMS64-NEXT:        SizeOfFunction: 0x18
; SYMS64-NEXT:        SymbolIndexOfNextBeyond: [[#IND+4]]
; SYMS64-NEXT:        Auxiliary Type: AUX_EXCEPT (0xFF)
; SYMS64-NEXT:      }
; SYMS64-NEXT:      Function Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+2]]
; SYMS64-NEXT:        SizeOfFunction: 0x18
; SYMS64-NEXT:        PointerToLineNum: 0x0
; SYMS64-NEXT:        SymbolIndexOfNextBeyond: [[#IND+4]]
; SYMS64-NEXT:        Auxiliary Type: AUX_FCN (0xFE)
; SYMS64-NEXT:      }
; SYMS64-NEXT:      CSECT Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+3]]
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
; SYMS64-NEXT:      Index: [[#IND+4]]
; SYMS64-NEXT:      Name: .test__trap_annotation_debug
; SYMS64-NEXT:      Value (RelocatableAddress): 0x28
; SYMS64-NEXT:      Section: .text
; SYMS64-NEXT:      Type: 0x0
; SYMS64-NEXT:      StorageClass: C_EXT (0x2)
; SYMS64-NEXT:      NumberOfAuxEntries: 3
; SYMS64-NEXT:      Exception Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+5]]
; SYMS64-NEXT:        OffsetToExceptionTable: 0x3AC
; SYMS64-NEXT:        SizeOfFunction: 0x68
; SYMS64-NEXT:        SymbolIndexOfNextBeyond: [[#IND+8]]
; SYMS64-NEXT:        Auxiliary Type: AUX_EXCEPT (0xFF)
; SYMS64-NEXT:      }
; SYMS64-NEXT:      Function Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+6]]
; SYMS64-NEXT:        SizeOfFunction: 0x68
; SYMS64-NEXT:        PointerToLineNum: 0x0
; SYMS64-NEXT:        SymbolIndexOfNextBeyond: [[#IND+8]]
; SYMS64-NEXT:        Auxiliary Type: AUX_FCN (0xFE)
; SYMS64-NEXT:      }
; SYMS64-NEXT:      CSECT Auxiliary Entry {
; SYMS64-NEXT:        Index: [[#IND+7]]
; SYMS64-NEXT:        ContainingCsectSymbolIndex: [[#IND-2]]
; SYMS64-NEXT:        ParameterHashIndex: 0x0
; SYMS64-NEXT:        TypeChkSectNum: 0x0
; SYMS64-NEXT:        SymbolAlignmentLog2: 0
; SYMS64-NEXT:        SymbolType: XTY_LD (0x2)
; SYMS64-NEXT:        StorageMappingClass: XMC_PR (0x0)
; SYMS64-NEXT:        Auxiliary Type: AUX_CSECT (0xFB)
; SYMS64-NEXT:      }
; SYMS64-NEXT:    }
