; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/only-needed-debug-metadata.ll -o %t2.bc

; Without -only-needed, we need to link in both DISubprogram.
; RUN: llvm-link -S %t2.bc %t.bc | FileCheck %s
; CHECK: distinct !DISubprogram(name: "foo"
; CHECK: distinct !DISubprogram(name: "unused"

; With -only-needed, we only need to link in foo's DISubprogram.
; RUN: llvm-link -S -only-needed %t2.bc %t.bc | FileCheck %s -check-prefix=ONLYNEEDED
; ONLYNEEDED: distinct !DISubprogram(name: "foo"
; ONLYNEEDED-NOT: distinct !DISubprogram(name: "unused"

source_filename = "test/Linker/only-needed-debug-metadata.ll"

@X = global i32 5, !dbg !0
@U = global i32 6, !dbg !6
@U_linkonce = linkonce_odr hidden global i32 6

define i32 @foo() !dbg !12 {
  ret i32 7, !dbg !17
}

define i32 @unused() !dbg !18 {
  ret i32 8, !dbg !21
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "X", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.8.0 (trunk 251407) (llvm/trunk 251401)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "linkused2.c", directory: "/usr/local/google/home/tejohnson/llvm/tmp")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "U", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.8.0 (trunk 251407) (llvm/trunk 251401)"}
!12 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 4, type: !13, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!8, !8}
!15 = !{!16}
!16 = !DILocalVariable(name: "x", arg: 1, scope: !12, file: !3, line: 4, type: !8)
!17 = !DILocation(line: 4, column: 13, scope: !12)
!18 = distinct !DISubprogram(name: "unused", scope: !3, file: !3, line: 8, type: !19, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !2, retainedNodes: !4)
!19 = !DISubroutineType(types: !20)
!20 = !{!8}
!21 = !DILocation(line: 9, column: 3, scope: !18)

