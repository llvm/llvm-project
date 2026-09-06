; AIX and zOS don't support DWARF 6 DW_AT_language_name
; XFAIL: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump -debug-info - \
; RUN:   | FileCheck %s --implicit-check-not=DW_AT_identifier_case

; Test that DW_AT_identifier_case is emitted for case-insensitive languages
; (Fortran) and omitted for case-sensitive languages (C++).

; Fortran via old-style language field
; CHECK:      DW_AT_language (DW_LANG_Fortran90)
; CHECK-NEXT: DW_AT_identifier_case (DW_ID_case_insensitive)

; C++ via old-style language field — no DW_AT_identifier_case emitted
; CHECK:      DW_AT_language (DW_LANG_C_plus_plus)

; Fortran via sourceLanguageName
; CHECK:      DW_AT_language_name (DW_LNAME_Fortran)
; CHECK:      DW_AT_identifier_case (DW_ID_case_insensitive)

; C++ via sourceLanguageName — no DW_AT_identifier_case emitted
; CHECK:      DW_AT_language_name (DW_LNAME_C_plus_plus)

@x = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2, !8, !9, !10}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!3 = !DIFile(filename: "test.f90", directory: "/tmp")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!9 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_Fortran, sourceLanguageVersion: 1990, file: !3, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!10 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 201100, file: !3, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
