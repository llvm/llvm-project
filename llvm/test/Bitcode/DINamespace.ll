; RUN: llvm-dis %s.bc -o - | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@_ZN1N1iE = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DIGlobalVariable(name: "i", linkageName: "_ZN1N1iE", scope: !1, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
; Test bitcode upgrade for DINamespace without an exportSymbols field.
; CHECK: !DINamespace(name: "N", scope: null)
!1 = !DINamespace(name: "N", scope: null)
!2 = !DIFile(filename: "dinamespace.cpp", directory: "/")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 4.0.0 (trunk 283228) (llvm/trunk 283225)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, globals: !6)
!5 = !{}
!6 = !{!0}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 4.0.0 (trunk 283228) (llvm/trunk 283225)"}
