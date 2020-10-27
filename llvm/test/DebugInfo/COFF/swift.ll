; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; ASM:      .short  4412                    # Record kind: S_COMPILE3
; ASM-NEXT: .long   83                      # Flags and language
; ASM-NEXT: .short  208                     # CPUType

; OBJ-LABEL: Compile3Sym {
; OBJ-NEXT:    Kind: S_COMPILE3 (0x113C)
; OBJ-NEXT:    Language: Swift (0x53)
; OBJ-NEXT:    Flags [ (0x0)
; OBJ-NEXT:    ]
; OBJ-NEXT:    Machine: X64 (0xD0)
; OBJ-NEXT:    FrontendVersion: {{[0-9.]*}}
; OBJ-NEXT:    BackendVersion: {{[0-9.]*}}
; OBJ-NEXT:    VersionName: Apple Swift version 5.0 (swiftlang-1001.0.45.7 clang-1001.0.37.7)
; OBJ-NEXT:  }


; ModuleID = 't.c'
source_filename = "t.swift"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

define void @f() !dbg !8 {
entry:
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Apple Swift version 5.0 (swiftlang-1001.0.45.7 clang-1001.0.37.7)", isOptimized: false, runtimeVersion: 5, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.d", directory: "asdf")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"Swift Version", i32 6}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 1, column: 11, scope: !8)
