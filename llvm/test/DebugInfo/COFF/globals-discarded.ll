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

@x = global i32 42

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36, !37}
!llvm.ident = !{!38}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 272215) (llvm/trunk 272226)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3)
!1 = !DIFile(filename: "t.c", directory: "foo")
!2 = !{}
!3 = !{!4, !6}
!4 = distinct !DIGlobalVariable(name: "_OptionsStorage", scope: !0, file: !1, line: 3, type: !5, isLocal: true, isDefinition: true)
!5 = !DIBasicType(name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!6 = distinct !DIGlobalVariable(name: "x", scope: !0, file: !1, line: 4, type: !5, isLocal: true, isDefinition: true, variable: i32* @x)

!35 = !{i32 2, !"CodeView", i32 1}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !{i32 1, !"PIC Level", i32 2}
!38 = !{!"clang version 3.9.0 (trunk 272215) (llvm/trunk 272226)"}
