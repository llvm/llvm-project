; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/entry_counts.prof -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/entry_counts.prof -S | FileCheck %s

; According to the profile, function empty() was called 13,293 times.
; CHECK: {{.*}} = !{!"function_entry_count", i64 13294}

define void @empty() #0 !dbg !4 {
entry:
  ret void, !dbg !9
}

; This function does not have profile, check if function_entry_count is -1
; CHECK: {{.*}} = !{!"function_entry_count", i64 -1}
define void @no_profile() #0 {
entry:
  ret void
}

attributes #0 = {"use-sample-profile"}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 237249) (llvm/trunk 237261)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "entry_counts.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "empty", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 3.7.0 (trunk 237249) (llvm/trunk 237261)"}
!9 = !DILocation(line: 1, column: 15, scope: !4)
