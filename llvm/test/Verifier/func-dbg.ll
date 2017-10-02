; RUN: llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s

define i32 @foo() !dbg !4 {
entry:
  ret i32 0, !dbg !6
}

define i32 @bar() !dbg !5 {
entry:
; CHECK: !dbg attachment points at wrong subprogram for function
  ret i32 0, !dbg !6
}

; CHECK: warning: ignoring invalid debug info
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "dwarf-test.c", directory: "test")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !0, isDefinition: true, unit: !0)
!5 = distinct !DISubprogram(name: "bar", scope: !0, isDefinition: true, unit: !0)
!6 = !DILocation(line: 7, scope: !4)
!7 = !{i32 2, !"Dwarf Version", i32 3}
!8 = !{i32 1, !"Debug Info Version", i32 3}
