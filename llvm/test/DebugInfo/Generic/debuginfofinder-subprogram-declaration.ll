; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; The declaration is only reachable from the definition's declaration: field.

; CHECK-COUNT-2: Subprogram: foo from /tmp/decl.cpp:3 ('_ZN1A3fooEv')

define i32 @_ZN1A3fooEv() !dbg !4 {
entry:
  ret i32 0, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "decl.cpp", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = distinct !DISubprogram(name: "foo", linkageName: "_ZN1A3fooEv", scope: !5, file: !1, line: 3, type: !3, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !6)
!5 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !1, line: 1, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS1A")
!6 = !DISubprogram(name: "foo", linkageName: "_ZN1A3fooEv", scope: !5, file: !1, line: 3, type: !3, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!7 = !{}
!9 = !DILocation(line: 3, column: 1, scope: !4)
