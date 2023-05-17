; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj %s -o - \
; RUN:   | llvm-dwarfdump -debug-line - | FileCheck %s --check-prefixes=CHECK

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj -dwarf64 %s -o - \
; RUN:   | llvm-dwarfdump -debug-line - | FileCheck %s --check-prefixes=CHECK64

; CHECK: file format aix5coff64-rs6000
; CHECK: format: DWARF32

; CHECK64: file format aix5coff64-rs6000
; CHECK64: format: DWARF64

source_filename = "1.c"
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32"

@foo = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "1.c", directory: "llvm-project")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 3}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 2}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!"clang version 17.0.0"}
