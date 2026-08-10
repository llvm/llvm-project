; This test checks that strip-dead-debug-info pass deletes debug compile units
; if functions from those units are absent in that module.

; RUN: opt -passes='strip-dead-debug-info,verify' %s -S | FileCheck %s

; CHECK: !llvm.dbg.cu = !{!{{[0-9]+}}, !{{[0-9]+}}}
; CHECK-COUNT-2: !DICompileUnit

;target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
;target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define void @dev_func1() #0 !dbg !13 {
  ret void, !dbg !14
}

; Function Attrs: nounwind
define void @dev_func2() #0 !dbg !15 {
  ret void, !dbg !16
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0, !3, !5}
!llvm.module.flags = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "dev_func1.cpp", directory: "/home/user/test")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !2, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "dev_func2.cpp", directory: "/home/user/test")
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !6, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, imports: !7, splitDebugInlining: false, nameTableKind: None)
!6 = !DIFile(filename: "dev_func3.cpp", directory: "/home/user/test")
!7 = !{!8}
!8 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !6, entity: !9, file: !6, line: 129)
!9 = distinct !DISubprogram(name: "dev_func3", linkageName: "dev_func3", scope: !6, file: !6, line: 96, type: !10, scopeLine: 96, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = distinct !DISubprogram(name: "dev_func1", linkageName: "dev_func1", scope: !1, file: !1, line: 22, type: !10, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 29, column: 5, scope: !13)
!15 = distinct !DISubprogram(name: "dev_func2", linkageName: "dev_func2", scope: !4, file: !4, line: 22, type: !10, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !2)
!16 = !DILocation(line: 29, column: 5, scope: !15)
