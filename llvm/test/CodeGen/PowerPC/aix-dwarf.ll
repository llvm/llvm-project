
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers %t.o | FileCheck %s --check-prefixes=SEC,SEC32
; RUN: llvm-objdump -r %t.o | FileCheck %s --check-prefix=RELO

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --section-headers %t64.o | FileCheck %s --check-prefixes=SEC,SEC64
; RUN: llvm-objdump -r %t64.o | FileCheck %s --check-prefix=RELO64

; This file is copied from test/DebugInfo/XCOFF/empty.ll.
; In this test, we focus on XCOFF related formats, like section headers,
; relocation entries.

source_filename = "1.c"
target datalayout = "E-m:a-p:32:32-i64:64-n32"

; Function Attrs: noinline nounwind optnone
define i32 @main() #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "1.c", directory: "debug")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 12.0.0"}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 3, column: 3, scope: !8)

; SEC:       Sections [
; SEC-NEXT:    Section {
; SEC-NEXT:      Index: 1
; SEC-NEXT:      Name: .text
; SEC-NEXT:      PhysicalAddress: 0x0
; SEC-NEXT:      VirtualAddress: 0x0
; SEC-NEXT:      Size: 0x28
; SEC32-NEXT:    RawDataOffset: 0xDC
; SEC64-NEXT:    RawDataOffset: 0x180
; SEC-NEXT:      RelocationPointer: 0x0
; SEC-NEXT:      LineNumberPointer: 0x0
; SEC-NEXT:      NumberOfRelocations: 0
; SEC-NEXT:      NumberOfLineNumbers: 0
; SEC-NEXT:      Type: STYP_TEXT (0x20)
; SEC-NEXT:    }
; SEC-NEXT:    Section {
; SEC-NEXT:      Index: 2
; SEC-NEXT:      Name: .data
; SEC-NEXT:      PhysicalAddress: 0x28
; SEC-NEXT:      VirtualAddress: 0x28
; SEC32-NEXT:    Size: 0xC
; SEC32-NEXT:    RawDataOffset: 0x104
; SEC32-NEXT:    RelocationPointer: 0x1F4
; SEC64-NEXT:    Size: 0x18
; SEC64-NEXT:    RawDataOffset: 0x1A8
; SEC64-NEXT:    RelocationPointer: 0x2C8
; SEC-NEXT:      LineNumberPointer: 0x0
; SEC-NEXT:      NumberOfRelocations: 2
; SEC-NEXT:      NumberOfLineNumbers: 0
; SEC-NEXT:      Type: STYP_DATA (0x40)
; SEC-NEXT:    }
; SEC-NEXT:    Section {
; SEC-NEXT:      Index: 3
; SEC-NEXT:      Name: .dwabrev
; SEC-NEXT:      PhysicalAddress: 0x0
; SEC-NEXT:      VirtualAddress: 0x0
; SEC-NEXT:      Size: 0x36
; SEC32-NEXT:    RawDataOffset: 0x11C
; SEC64-NEXT:    RawDataOffset: 0x1C0
; SEC-NEXT:      RelocationPointer: 0x0
; SEC-NEXT:      LineNumberPointer: 0x0
; SEC-NEXT:      NumberOfRelocations: 0
; SEC-NEXT:      NumberOfLineNumbers: 0
; SEC-NEXT:      Type: STYP_DWARF (0x10)
; SEC-NEXT:    }
; SEC-NEXT:    Section {
; SEC-NEXT:      Index: 4
; SEC-NEXT:      Name: .dwinfo
; SEC-NEXT:      PhysicalAddress: 0x0
; SEC-NEXT:      VirtualAddress: 0x0
; SEC32-NEXT:    Size: 0x57
; SEC32-NEXT:    RawDataOffset: 0x15C
; SEC32-NEXT:    RelocationPointer: 0x208
; SEC64-NEXT:    Size: 0x6F
; SEC64-NEXT:    RawDataOffset: 0x200
; SEC64-NEXT:    RelocationPointer: 0x2E4
; SEC-NEXT:      LineNumberPointer: 0x0
; SEC-NEXT:      NumberOfRelocations: 4
; SEC-NEXT:      NumberOfLineNumbers: 0
; SEC-NEXT:      Type: STYP_DWARF (0x10)
; SEC-NEXT:    }
; SEC-NEXT:    Section {
; SEC-NEXT:      Index: 5
; SEC-NEXT:      Name: .dwline
; SEC-NEXT:      PhysicalAddress: 0x0
; SEC-NEXT:      VirtualAddress: 0x0
; SEC32-NEXT:    Size: 0x36
; SEC32-NEXT:    RawDataOffset: 0x1BC
; SEC32-NEXT:    RelocationPointer: 0x230
; SEC64-NEXT:    Size: 0x46
; SEC64-NEXT:    RawDataOffset: 0x280
; SEC64-NEXT:    RelocationPointer: 0x31C
; SEC-NEXT:      LineNumberPointer: 0x0
; SEC-NEXT:      NumberOfRelocations: 1
; SEC-NEXT:      NumberOfLineNumbers: 0
; SEC-NEXT:      Type: STYP_DWARF (0x10)
; SEC-NEXT:    }
; SEC-NEXT:  ]

; RELO:      RELOCATION RECORDS FOR [.dwinfo]:
; RELO-NEXT:  OFFSET   TYPE                     VALUE
; RELO-NEXT:  00000006 R_POS                    .dwabrev
; RELO-NEXT:  00000027 R_POS                    .dwline
; RELO-NEXT:  00000009 R_POS                    .text
; RELO-NEXT:  0000003a R_POS                    .text
; RELO:       RELOCATION RECORDS FOR [.dwline]:
; RELO-NEXT:  OFFSET   TYPE                     VALUE
; RELO-NEXT:  00000000 R_POS                    .text

; RELO64:      RELOCATION RECORDS FOR [.dwinfo]:
; RELO64-NEXT: OFFSET           TYPE                     VALUE
; RELO64-NEXT: 000000000000000e R_POS                    .dwabrev
; RELO64-NEXT: 000000000000000b R_POS                    .dwline
; RELO64-NEXT: 0000000000000041 R_POS                    .text
; RELO64-NEXT: 000000000000004e R_POS                    .text
; RELO64:      RELOCATION RECORDS FOR [.dwline]:
; RELO64-NEXT: OFFSET           TYPE                     VALUE
; RELO64-NEXT: 000000000000000c R_POS                    .text
