; RUN: not llc -mtriple=bpfel -filetype=obj %s -o %t 2>&1 | FileCheck %s --implicit-check-not=error:
;
; Source code:
;   __attribute__((common)) typeof(nullptr) a, b;
; Compilation flag:
;   clang -target bpfel -std=c23 -O2 -g -S -emit-llvm globals.c
;
; Neither common global gets a BTF VAR record. Visiting their shared type
; must still diagnose it exactly once.
; CHECK: error: unsupported BTF type 'decltype(nullptr)'

@a = common global ptr null, align 8, !dbg !0
@b = common global ptr null, align 8, !dbg !2

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !4, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = distinct !DIGlobalVariable(name: "b", scope: !4, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true)
!4 = distinct !DICompileUnit(language: DW_LANG_C11, file: !5, emissionKind: FullDebug, globals: !{!0, !2})
!5 = !DIFile(filename: "globals.c", directory: "/")
!6 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!7 = !{i32 2, !"Debug Info Version", i32 3}
