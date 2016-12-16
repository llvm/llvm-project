; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s

; Just because there are no scopes/locations on any instructions in the
; function doesn't mean we can't describe the address range of the function.
; Check that we do that

; CHECK: DW_TAG_subprogram
; CHECK-NOT: TAG
; CHECK:   DW_AT_low_pc

; Function Attrs: nounwind uwtable
define void @f() #0 !dbg !6 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 289692) (llvm/trunk 289697)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "noscopes.c", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 (trunk 289692) (llvm/trunk 289697)"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 2, column: 1, scope: !6)

