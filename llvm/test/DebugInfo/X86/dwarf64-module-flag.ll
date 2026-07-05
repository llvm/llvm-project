; This checks that the debug info is generated in the 64-bit format if the
; module has the corresponding flag.

; RUN: llc -mtriple=x86_64 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info -debug-line %t | FileCheck %s

; CHECK:      Compile Unit: {{.*}} format = DWARF64
; CHECK:      debug_line[
; CHECK-NEXT: Line table prologue:
; CHECK-NEXT:   total_length:
; CHECK-NEXT:     format: DWARF64

; IR generated and reduced from:
; $ cat foo.c
; int foo;
; $ clang -g -gdwarf64 -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 7, !"DWARF64", i32 1}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 13.0.0"}
