; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s --check-prefix CHECK
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s --check-prefix CHECK

; RUN: llc -filetype=obj -mtriple powerpc-ibm-aix-xcoff < %s -o %t32.o
; RUN: llvm-readobj %t32.o --syms | FileCheck %s --check-prefix=OBJ32
; RUN: llc -filetype=obj -mtriple powerpc64-ibm-aix-xcoff < %s -o %t64.o
; RUN: llvm-readobj %t64.o --syms | FileCheck %s --check-prefix=OBJ64

%struct.small_struct = type { i16 }

@a = global %struct.small_struct zeroinitializer, align 2 #0
@b = global [2 x i16] zeroinitializer, align 2 #0

; Function Attrs: noinline
define i16 @foo() #1 {
entry:
  %0 = load i16, ptr @a, align 2
  %1 = load i16, ptr @b, align 2
  %add = add nsw i16 %0, %1
  ret i16 %add
}

attributes #0 = { "toc-data" }
attributes #1 = { noinline }

; CHECK:      .toc
; CHECK-NEXT: .csect a[TD],2
; CHECK-NEXT: .globl    a[TD]                           # @a
; CHECK-NEXT: .align    1
; CHECK-NEXT: .space    2
; CHECK-NEXT: .csect b[TD],2
; CHECK-NEXT: .globl    b[TD]                           # @b
; CHECK-NEXT: .align    1
; CHECK-NEXT: .space    4

; OBJ32:  Symbol {
; OBJ32:    Name: a
; OBJ32-NEXT:    Value (RelocatableAddress): 0x3C
; OBJ32-NEXT:    Section: .data
; OBJ32-NEXT:    Type: 0x0
; OBJ32-NEXT:    StorageClass: C_EXT (0x2)
; OBJ32-NEXT:    NumberOfAuxEntries: 1
; OBJ32-NEXT:    CSECT Auxiliary Entry {
; OBJ32-NEXT:      Index: {{[0-9]+}}
; OBJ32-NEXT:      SectionLen: 2
; OBJ32-NEXT:      ParameterHashIndex: 0x0
; OBJ32-NEXT:      TypeChkSectNum: 0x0
; OBJ32-NEXT:      SymbolAlignmentLog2: 2
; OBJ32-NEXT:      SymbolType: XTY_SD (0x1)
; OBJ32-NEXT:      StorageMappingClass: XMC_TD (0x10)
; OBJ32-NEXT:      StabInfoIndex: 0x0
; OBJ32-NEXT:      StabSectNum: 0x0
; OBJ32-NEXT:    }
; OBJ32-NEXT:  }
; OBJ32-NEXT:  Symbol {
; OBJ32:    Name: b
; OBJ32-NEXT:    Value (RelocatableAddress): 0x40
; OBJ32-NEXT:    Section: .data
; OBJ32-NEXT:    Type: 0x0
; OBJ32-NEXT:    StorageClass: C_EXT (0x2)
; OBJ32-NEXT:    NumberOfAuxEntries: 1
; OBJ32-NEXT:    CSECT Auxiliary Entry {
; OBJ32-NEXT:      Index: {{[0-9]+}}
; OBJ32-NEXT:      SectionLen: 4
; OBJ32-NEXT:      ParameterHashIndex: 0x0
; OBJ32-NEXT:      TypeChkSectNum: 0x0
; OBJ32-NEXT:      SymbolAlignmentLog2: 2
; OBJ32-NEXT:      SymbolType: XTY_SD (0x1)
; OBJ32-NEXT:      StorageMappingClass: XMC_TD (0x10)
; OBJ32-NEXT:      StabInfoIndex: 0x0
; OBJ32-NEXT:      StabSectNum: 0x0
; OBJ32-NEXT:    }
; OBJ32-NEXT:  }

; OBJ64:  Symbol {
; OBJ64:    Name: a
; OBJ64-NEXT:    Value (RelocatableAddress): 0x48
; OBJ64-NEXT:    Section: .data
; OBJ64-NEXT:    Type: 0x0
; OBJ64-NEXT:    StorageClass: C_EXT (0x2)
; OBJ64-NEXT:    NumberOfAuxEntries: 1
; OBJ64-NEXT:    CSECT Auxiliary Entry {
; OBJ64-NEXT:      Index: {{[0-9]+}}
; OBJ64-NEXT:      SectionLen: 2
; OBJ64-NEXT:      ParameterHashIndex: 0x0
; OBJ64-NEXT:      TypeChkSectNum: 0x0
; OBJ64-NEXT:      SymbolAlignmentLog2: 2
; OBJ64-NEXT:      SymbolType: XTY_SD (0x1)
; OBJ64-NEXT:      StorageMappingClass: XMC_TD (0x10)
; OBJ64-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; OBJ64-NEXT:    }
; OBJ64-NEXT:  }
; OBJ64-NEXT:  Symbol {
; OBJ64:    Name: b
; OBJ64-NEXT:    Value (RelocatableAddress): 0x4C
; OBJ64-NEXT:    Section: .data
; OBJ64-NEXT:    Type: 0x0
; OBJ64-NEXT:    StorageClass: C_EXT (0x2)
; OBJ64-NEXT:    NumberOfAuxEntries: 1
; OBJ64-NEXT:    CSECT Auxiliary Entry {
; OBJ64-NEXT:      Index: {{[0-9]+}}
; OBJ64-NEXT:      SectionLen: 4
; OBJ64-NEXT:      ParameterHashIndex: 0x0
; OBJ64-NEXT:      TypeChkSectNum: 0x0
; OBJ64-NEXT:      SymbolAlignmentLog2: 2
; OBJ64-NEXT:      SymbolType: XTY_SD (0x1)
; OBJ64-NEXT:      StorageMappingClass: XMC_TD (0x10)
; OBJ64-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; OBJ64-NEXT:    }
; OBJ64-NEXT:  }
