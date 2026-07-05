; Test that LLParser rejects DISubprogram with missing or null 'type' field.
; These are parse-time errors, not verifier errors. See issue #186557.
;
; RUN: split-file %s %t
; RUN: not llvm-as %t/missing.ll -disable-output 2>&1 | FileCheck %s --check-prefix=MISSING
; RUN: not llvm-as %t/null.ll    -disable-output 2>&1 | FileCheck %s --check-prefix=NULL
;
; MISSING: missing required field 'type'
; NULL: 'type' cannot be null

;--- missing.ll
define void @f() !dbg !4 { ret void }
!llvm.dbg.cu       = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "x.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1,
     scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)

;--- null.ll
define void @f() !dbg !4 { ret void }
!llvm.dbg.cu       = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "x.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1,
     scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, type: null)
