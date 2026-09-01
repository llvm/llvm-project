; In the ThinLTO backend, LowerTypeTests renames the body of a function that is
; a CFI jump table member from foo to foo.cfi. Profiles are keyed by the
; original name, so the suffix must be ignored when looking up samples, also
; when it follows the suffix of a promoted local function.
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/cfi-suffix.prof -S | FileCheck %s

; CHECK: define void @foo.cfi() {{.*}} !prof ![[FOO:[0-9]+]]
define void @foo.cfi() #0 !dbg !4 {
entry:
  ret void, !dbg !9
}

; CHECK: define void @bar.llvm.1234.cfi() {{.*}} !prof ![[BAR:[0-9]+]]
define void @bar.llvm.1234.cfi() #0 !dbg !10 {
entry:
  ret void, !dbg !11
}

; CHECK: ![[FOO]] = !{!"function_entry_count", i64 1001}
; CHECK: ![[BAR]] = !{!"function_entry_count", i64 2001}

attributes #0 = {"use-sample-profile" "sample-profile-suffix-elision-policy"="selected"}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "cfi-suffix.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DILocation(line: 1, column: 15, scope: !4)
!10 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: false, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 3, column: 15, scope: !10)
