; This checks that .debug_info.dwo can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf-version=5 -dwarf64 -split-dwarf-file=foo.dwo -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; CHECK:      .debug_info.dwo contents:
; CHECK-NEXT: Compile Unit: {{.+}}, format = DWARF64, version = 0x0005, unit_type = DW_UT_split_compile, abbr_offset = 0x0000,

; IR generated and reduced from:
; $ cat foo.c
; int foo;
; $ clang -g -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
