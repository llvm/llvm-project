; REQUIRES: object-emission
; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

; CHECK: {{.*}}DW_TAG_variable
; CHECK-NEXT: {{.*}} DW_AT_const_value     (1)
; CHECK-NEXT: {{.*}} DW_AT_name    ("arg")

define void @test() !dbg !5
{
entry:
  call void @"llvm.dbg.value"(metadata i1 true, metadata !7, metadata !8), !dbg !6
  ret void, !dbg !6
}

declare void @"llvm.dbg.value"(metadata %".1", metadata %".2", metadata %".3")

!llvm.dbg.cu = !{ !2 }
!llvm.module.flags = !{ !9, !10 }

!1 = !DIFile(directory: "", filename: "test")
!2 = distinct !DICompileUnit(emissionKind: FullDebug, file: !1, isOptimized: false, language: DW_LANG_C_plus_plus, runtimeVersion: 0)
!3 = !DIBasicType(encoding: DW_ATE_boolean, name: "bool", size: 8)
!4 = !DISubroutineType(types: !{null})
!5 = distinct !DISubprogram(file: !1, isDefinition: true, isLocal: false, isOptimized: false, line: 5, linkageName: "test", name: "test", scope: !1, scopeLine: 5, type: !4, unit: !2)
!6 = !DILocation(column: 1, line: 5, scope: !5)
!7 = !DILocalVariable(arg: 0, file: !1, line: 5, name: "arg", scope: !5, type: !3)
!8 = !DIExpression()
!9 = !{ i32 2, !"Dwarf Version", i32 4 }
!10 = !{ i32 2, !"Debug Info Version", i32 3 }
