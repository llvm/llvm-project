; REQUIRES: object-emission
; RUN: %llc_dwarf -O0 -filetype=obj < %S/gmlt_profiling.ll | llvm-dwarfdump - | FileCheck %S/gmlt_profiling.ll

; CHECK: .debug_info
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "f1"
; With debug-info-for-profiling attribute, we need to emit decl_file and
; decl_line of the subprogram.
; CHECK-NEXT: DW_AT_decl_file
; CHECK-NEXT: DW_AT_decl_line

; Function Attrs: nounwind uwtable
define void @_Z2f1v() !dbg !4 {
entry:
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2, debugInfoForProfiling: true)
!1 = !DIFile(filename: "gmlt.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "f1", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "gmlt.cpp", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !2)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.6.0 "}
!13 = !DILocation(line: 1, column: 12, scope: !4)
