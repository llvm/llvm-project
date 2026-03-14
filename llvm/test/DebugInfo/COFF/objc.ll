; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; ASM:      .short  4412                    # Record kind: S_COMPILE3
; ASM-NEXT: .long   17                      # Flags and language

; OBJ:       Kind: S_COMPILE3 (0x113C)
; OBJ-NEXT:  Language: ObjC (0x11)

; ModuleID = 'objc.m'
source_filename = "objc.m"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Function Attrs: uwtable
define void @f() unnamed_addr #0 !dbg !5 {
entry:
  ret void, !dbg !9
}

attributes #0 = { uwtable "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!3}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"CodeView", i32 1}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !4, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project a8e9beca6bee1f248ef4be7892802c4d091b7fcb)", isOptimized: false, runtimeVersion: 1, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "objc.m", directory: "src", checksumkind: CSK_MD5, checksum: "e6ab1d5b7f82464c963a8522037dfa72")
!5 = distinct !DISubprogram(name: "f", scope: !4, file: !4, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{}
!9 = !DILocation(line: 1, scope: !5)
