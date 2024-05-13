; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; ASM:      .short  4412                    # Record kind: S_COMPILE3
; ASM-NEXT: .long   21                      # Flags and language
; ASM-NEXT: .short  208                     # CPUType

; OBJ-LABEL: Compile3Sym {
; OBJ-NEXT:    Kind: S_COMPILE3 (0x113C)
; OBJ-NEXT:    Language: Rust (0x15)
; OBJ-NEXT:    Flags [ (0x0)
; OBJ-NEXT:    ]
; OBJ-NEXT:    Machine: X64 (0xD0)
; OBJ-NEXT:    FrontendVersion: {{[0-9.]*}}
; OBJ-NEXT:    BackendVersion: {{[0-9.]*}}
; OBJ-NEXT:    VersionName: clang LLVM (rustc version 1.57.0 (f1edd0429 2021-11-29))
; OBJ-NEXT:  }


; ModuleID = 'main.a61fec89-cgu.0'
source_filename = "main.a61fec89-cgu.0"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Function Attrs: uwtable
define void @f() unnamed_addr #0 !dbg !6 {
start:
  ret void, !dbg !11
}

attributes #0 = { uwtable "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!3}

!0 = !{i32 7, !"PIC Level", i32 2}
!1 = !{i32 2, !"CodeView", i32 1}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !4, producer: "clang LLVM (rustc version 1.57.0 (f1edd0429 2021-11-29))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5)
!4 = !DIFile(filename: "main.rs", directory: "src")
!5 = !{}
!6 = distinct !DISubprogram(name: "f", scope: !8, file: !7, line: 13, type: !9, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, templateParams: !5, retainedNodes: !5)
!7 = !DIFile(filename: "main.rs", directory: "src", checksumkind: CSK_SHA1, checksum: "2ac9107db410c2ac03093f537ff521068091fb92")
!8 = !DINamespace(name: "main", scope: null)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 14, scope: !12)
!12 = !DILexicalBlockFile(scope: !6, file: !7, discriminator: 0)
