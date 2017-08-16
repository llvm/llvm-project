; RUN: opt < %s -simplifycfg -S | FileCheck %s

; Checks if the debug info is removed for the "select" instruction.
; CHECK: cmp {{.*}} !dbg
; CHECK-NOT: select {{.*}} !dbg
define i32 @min(i32 %a, i32 %b) {
entry:
  %cmp = icmp slt i32 %a, %b, !dbg !9
  br i1 %cmp, label %if.then, label %if.else, !dbg !10

if.then:
  br label %return, !dbg !11

if.else:
  br label %return, !dbg !12

return:
  %retval.0 = phi i32 [ %a, %if.then ], [ %b, %if.else ]
  ret i32 %retval.0, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 310792)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "min.cc", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 (trunk 310792)"}
!7 = distinct !DISubprogram(name: "min", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, column: 8, scope: !7)
!10 = !DILocation(line: 4, column: 6, scope: !7)
!11 = !DILocation(line: 5, column: 3, scope: !7)
!12 = !DILocation(line: 7, column: 3, scope: !7)
!13 = !DILocation(line: 9, column: 1, scope: !7)
