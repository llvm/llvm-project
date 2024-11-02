; This file tests TOC entry generation and undefined symbol generation.

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s | FileCheck --check-prefixes CHECK,CHECK32 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s 2>&1 | FileCheck --check-prefixes CHECK,CHECK64  %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --xcoff-traceback-table=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefixes=SYM,SYM32 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --xcoff-traceback-table=false -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --syms %t64.o | FileCheck --check-prefixes=SYM,SYM64 %s

@a = external global i32, align 4
@b = external global i64, align 8
@c = external global i16, align 2
@globa = common global i32 0, align 4

@ptr = internal global void (...)* null, align 4

declare void @foo()

define void @bar() {
  %1 = alloca i8*, align 8
  %2 = alloca i8*, align 8
  store i32 0, i32* @a, align 4
  store i64 0, i64* @b, align 8
  store i16 0, i16* @c, align 2
  store i32 0, i32* @globa, align 4
  store void (...)* bitcast (void ()* @bar to void (...)*), void (...)** @ptr, align 4
  store i8* bitcast (void ()* @foo to i8*), i8** %1, align 8
  store i8* bitcast (void ()* @foobar to i8*), i8** %2, align 8
  ret void
}

; We initialize a csect when we first reference an external global, so make sure we don't run into problems when we see it again.
define void @bar2() {
  store i32 0, i32* @a, align 4
  store i64 0, i64* @b, align 8
  store i16 0, i16* @c, align 2
  ret void
}

define void @foobar() {
  ret void
}

; Test tc entry assembly generation.

; CHECK-NOT: .comm a
; CHECK-NOT: .lcomm a
; CHECK-NOT: .comm b
; CHECK-NOT: .lcomm b
; CHECK-NOT: .comm c
; CHECK-NOT: .lcomm c
; CHECK: .comm globa[RW],4,2
; CHECK32: .lcomm ptr,4,ptr[BS],2
; CHECK64: .lcomm ptr,8,ptr[BS],2
; CHECK:      .toc
; CHECK-NEXT: L..C0:
; CHECK-NEXT: .tc   a[TC],a[UA]
; CHECK-NEXT: L..C1:
; CHECK-NEXT: .tc   b[TC],b[UA]
; CHECK-NEXT: L..C2:
; CHECK-NEXT: .tc   c[TC],c[UA]
; CHECK-NEXT: L..C3:
; CHECK-NEXT: .tc   globa[TC],globa[RW]
; CHECK-NEXT: L..C4:
; CHECK-NEXT: .tc   ptr[TC],ptr[BS]
; CHECK-NEXT: L..C5:
; CHECK-NEXT: .tc   bar[TC],bar[DS]
; CHECK-NEXT: L..C6:
; CHECK-NEXT: .tc   foo[TC],foo[DS]
; CHECK-NEXT: L..C7:
; CHECK-NEXT: .tc   foobar[TC],foobar[DS]


; Test undefined symbol generation.

; SYM:       Symbol {{[{][[:space:]] *}}Index: [[#UNDEF_INDX:]]{{[[:space:]] *}}Name: a
; SYM-NEXT:   Value (RelocatableAddress): 0x0
; SYM-NEXT:   Section: N_UNDEF
; SYM-NEXT:   Type: 0x0
; SYM-NEXT:   StorageClass: C_EXT (0x2)
; SYM-NEXT:   NumberOfAuxEntries: 1
; SYM-NEXT:   CSECT Auxiliary Entry {
; SYM-NEXT:     Index: [[#UNDEF_INDX+1]]
; SYM-NEXT:     SectionLen: 0
; SYM-NEXT:     ParameterHashIndex: 0x0
; SYM-NEXT:     TypeChkSectNum: 0x0
; SYM-NEXT:     SymbolAlignmentLog2: 0
; SYM-NEXT:     SymbolType: XTY_ER (0x0)
; SYM-NEXT:     StorageMappingClass: XMC_UA (0x4)
; SYM32-NEXT:   StabInfoIndex: 0x0
; SYM32-NEXT:   StabSectNum: 0x0
; SYM64-NEXT:   Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:   }
; SYM-NEXT: }
; SYM-NEXT: Symbol {
; SYM-NEXT:   Index: [[#UNDEF_INDX+2]]
; SYM-NEXT:   Name: b
; SYM-NEXT:   Value (RelocatableAddress): 0x0
; SYM-NEXT:   Section: N_UNDEF
; SYM-NEXT:   Type: 0x0
; SYM-NEXT:   StorageClass: C_EXT (0x2)
; SYM-NEXT:   NumberOfAuxEntries: 1
; SYM-NEXT:   CSECT Auxiliary Entry {
; SYM-NEXT:     Index: [[#UNDEF_INDX+3]]
; SYM-NEXT:     SectionLen: 0
; SYM-NEXT:     ParameterHashIndex: 0x0
; SYM-NEXT:     TypeChkSectNum: 0x0
; SYM-NEXT:     SymbolAlignmentLog2: 0
; SYM-NEXT:     SymbolType: XTY_ER (0x0)
; SYM-NEXT:     StorageMappingClass: XMC_UA (0x4)
; SYM32-NEXT:   StabInfoIndex: 0x0
; SYM32-NEXT:   StabSectNum: 0x0
; SYM64-NEXT:   Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:   }
; SYM-NEXT: }
; SYM-NEXT: Symbol {
; SYM-NEXT:   Index: [[#UNDEF_INDX+4]]
; SYM-NEXT:   Name: c
; SYM-NEXT:   Value (RelocatableAddress): 0x0
; SYM-NEXT:   Section: N_UNDEF
; SYM-NEXT:   Type: 0x0
; SYM-NEXT:   StorageClass: C_EXT (0x2)
; SYM-NEXT:   NumberOfAuxEntries: 1
; SYM-NEXT:   CSECT Auxiliary Entry {
; SYM-NEXT:     Index: [[#UNDEF_INDX+5]]
; SYM-NEXT:     SectionLen: 0
; SYM-NEXT:     ParameterHashIndex: 0x0
; SYM-NEXT:     TypeChkSectNum: 0x0
; SYM-NEXT:     SymbolAlignmentLog2: 0
; SYM-NEXT:     SymbolType: XTY_ER (0x0)
; SYM-NEXT:     StorageMappingClass: XMC_UA (0x4)
; SYM32-NEXT:   StabInfoIndex: 0x0
; SYM32-NEXT:   StabSectNum: 0x0
; SYM64-NEXT:   Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:   }
; SYM-NEXT: }
; SYM-NEXT: Symbol {
; SYM-NEXT:   Index: [[#UNDEF_INDX+6]]
; SYM-NEXT:   Name: .foo
; SYM-NEXT:   Value (RelocatableAddress): 0x0
; SYM-NEXT:   Section: N_UNDEF
; SYM-NEXT:   Type: 0x0
; SYM-NEXT:   StorageClass: C_EXT (0x2)
; SYM-NEXT:   NumberOfAuxEntries: 1
; SYM-NEXT:   CSECT Auxiliary Entry {
; SYM-NEXT:     Index: [[#UNDEF_INDX+7]]
; SYM-NEXT:     SectionLen: 0
; SYM-NEXT:     ParameterHashIndex: 0x0
; SYM-NEXT:     TypeChkSectNum: 0x0
; SYM-NEXT:     SymbolAlignmentLog2: 0
; SYM-NEXT:     SymbolType: XTY_ER (0x0)
; SYM-NEXT:     StorageMappingClass: XMC_PR (0x0)
; SYM32-NEXT:   StabInfoIndex: 0x0
; SYM32-NEXT:   StabSectNum: 0x0
; SYM64-NEXT:   Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:   }
; SYM-NEXT: }
; SYM-NEXT: Symbol {
; SYM-NEXT:   Index: [[#UNDEF_INDX+8]]
; SYM-NEXT:   Name: foo
; SYM-NEXT:   Value (RelocatableAddress): 0x0
; SYM-NEXT:   Section: N_UNDEF
; SYM-NEXT:   Type: 0x0
; SYM-NEXT:   StorageClass: C_EXT (0x2)
; SYM-NEXT:   NumberOfAuxEntries: 1
; SYM-NEXT:   CSECT Auxiliary Entry {
; SYM-NEXT:     Index: [[#UNDEF_INDX+9]]
; SYM-NEXT:     SectionLen: 0
; SYM-NEXT:     ParameterHashIndex: 0x0
; SYM-NEXT:     TypeChkSectNum: 0x0
; SYM-NEXT:     SymbolAlignmentLog2: 0
; SYM-NEXT:     SymbolType: XTY_ER (0x0)
; SYM-NEXT:     StorageMappingClass: XMC_DS (0xA)
; SYM32-NEXT:   StabInfoIndex: 0x0
; SYM32-NEXT:   StabSectNum: 0x0
; SYM64-NEXT:   Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:   }
; SYM-NEXT: }

; Test TOC entry symbol generation.

; SYM:       Symbol {{[{][[:space:]] *}}Index: [[#TOC_INDX:]]{{[[:space:]] *}}Name: TOC
; SYM32-NEXT:  Value (RelocatableAddress): 0xA8
; SYM64-NEXT:  Value (RelocatableAddress): 0xC0
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+1]]
; SYM-NEXT:      SectionLen: 0
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 2
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC0 (0xF)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM-NEXT:  Symbol {
; SYM-NEXT:    Index: [[#TOC_INDX+2]]
; SYM-NEXT:    Name: a
; SYM32-NEXT:  Value (RelocatableAddress): 0xA8
; SYM64-NEXT:  Value (RelocatableAddress): 0xC0
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+3]]
; SYM32-NEXT:    SectionLen: 4
; SYM64-NEXT:    SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM32-NEXT:    SymbolAlignmentLog2: 2
; SYM64-NEXT:    SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM-NEXT:  Symbol {
; SYM-NEXT:    Index: [[#TOC_INDX+4]]
; SYM-NEXT:    Name: b
; SYM32-NEXT:  Value (RelocatableAddress): 0xAC
; SYM64-NEXT:  Value (RelocatableAddress): 0xC8
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+5]]
; SYM32-NEXT:    SectionLen: 4
; SYM64-NEXT:    SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM32-NEXT:    SymbolAlignmentLog2: 2
; SYM64-NEXT:    SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM-NEXT:  Symbol {
; SYM-NEXT:    Index: [[#TOC_INDX+6]]
; SYM-NEXT:    Name: c
; SYM32-NEXT:  Value (RelocatableAddress): 0xB0
; SYM64-NEXT:  Value (RelocatableAddress): 0xD0
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+7]]
; SYM32-NEXT:    SectionLen: 4
; SYM64-NEXT:    SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM32-NEXT:    SymbolAlignmentLog2: 2
; SYM64-NEXT:    SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM-NEXT:  Symbol {
; SYM-NEXT:    Index: [[#TOC_INDX+8]]
; SYM-NEXT:    Name: globa
; SYM32-NEXT:  Value (RelocatableAddress): 0xB4
; SYM64-NEXT:  Value (RelocatableAddress): 0xD8
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+9]]
; SYM32-NEXT:    SectionLen: 4
; SYM64-NEXT:    SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM32-NEXT:    SymbolAlignmentLog2: 2
; SYM64-NEXT:    SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM-NEXT:  Symbol {
; SYM-NEXT:    Index: [[#TOC_INDX+10]]
; SYM-NEXT:    Name: ptr
; SYM32-NEXT:  Value (RelocatableAddress): 0xB8
; SYM64-NEXT:  Value (RelocatableAddress): 0xE0
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+11]]
; SYM32-NEXT:    SectionLen: 4
; SYM64-NEXT:    SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM32-NEXT:    SymbolAlignmentLog2: 2
; SYM64-NEXT:    SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM-NEXT:  Symbol {
; SYM-NEXT:    Index: [[#TOC_INDX+12]]
; SYM-NEXT:    Name: bar
; SYM32-NEXT:  Value (RelocatableAddress): 0xBC
; SYM64-NEXT:  Value (RelocatableAddress): 0xE8
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+13]]
; SYM32-NEXT:    SectionLen: 4
; SYM64-NEXT:    SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM32-NEXT:    SymbolAlignmentLog2: 2
; SYM64-NEXT:    SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM-NEXT:  Symbol {
; SYM-NEXT:    Index: [[#TOC_INDX+14]]
; SYM-NEXT:    Name: foo
; SYM32-NEXT:  Value (RelocatableAddress): 0xC0
; SYM64-NEXT:  Value (RelocatableAddress): 0xF0
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+15]]
; SYM32-NEXT:    SectionLen: 4
; SYM64-NEXT:    SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM32-NEXT:    SymbolAlignmentLog2: 2
; SYM64-NEXT:    SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM-NEXT:  Symbol {
; SYM-NEXT:    Index: [[#TOC_INDX+16]]
; SYM-NEXT:    Name: foobar
; SYM32-NEXT:  Value (RelocatableAddress): 0xC4
; SYM64-NEXT:  Value (RelocatableAddress): 0xF8
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: [[#TOC_INDX+17]]
; SYM32-NEXT:    SectionLen: 4
; SYM64-NEXT:    SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM32-NEXT:    SymbolAlignmentLog2: 2
; SYM64-NEXT:    SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:    StabInfoIndex: 0x0
; SYM32-NEXT:    StabSectNum: 0x0
; SYM64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
