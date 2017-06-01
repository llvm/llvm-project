; RUN: llc < %s | FileCheck %s

; This tests that we don't emit information about globals that were discarded
; during optimization. We should only see one global symbol record.

; CHECK: .short  4364                    # Record kind: S_LDATA32
; CHECK: .long   117                     # Type
; CHECK: .secrel32       x               # DataOffset
; CHECK: .secidx x                       # Segment
; CHECK: .asciz  "x"                     # Name
; CHECK-NOT: S_GDATA32

; ModuleID = 't.ii'
source_filename = "t.ii"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.0"

@x = global i32 42, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 4, type: !8, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (trunk 272215) (llvm/trunk 272226)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5)
!3 = !DIFile(filename: "t.c", directory: "foo")
!4 = !{}
!5 = !{!6, !0}
!6 = distinct !DIGlobalVariableExpression(var: !7)
!7 = !DIGlobalVariable(name: "_OptionsStorage", scope: !2, file: !3, line: 3, type: !8, isLocal: true, isDefinition: true)
!8 = !DIBasicType(name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!9 = !{i32 2, !"CodeView", i32 1}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"PIC Level", i32 2}
!12 = !{!"clang version 3.9.0 (trunk 272215) (llvm/trunk 272226)"}

