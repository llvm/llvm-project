; RUN: opt -S -passes=strip-nonlinetable-debuginfo %s -o - | FileCheck %s
; Test that subroutine types are downgraded to (void)().
define internal i32 @"__hidden#2878_"() #0 !dbg !12 {
  ret i32 0, !dbg !634
}
!llvm.dbg.cu = !{!18}
!llvm.module.flags = !{!482}
!5 = !{}
!2 = !{!12}
; CHECK-NOT: DICompositeType
; CHECK: distinct !DISubprogram(name: "f", {{.*}}, type: ![[FNTY:[0-9]+]]
; CHECK: ![[FNTY]] = !DISubroutineType(types: ![[VOID:[0-9]+]])
; CHECK: ![[VOID]] = !{}
; CHECK-NOT: DICompositeType
!12 = distinct !DISubprogram(name: "f", scope: !16, file: !16, line: 133, type: !13, isLocal: true, isDefinition: true, scopeLine: 133, flags: DIFlagPrototyped, isOptimized: true, unit: !18, retainedNodes: !5)
!13 = !DISubroutineType(types: !14)
!14 = !{!17}
!16 = !DIFile(filename: "f.m", directory: "/")
!17 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e", scope: !18, file: !16, line: 13, size: 32, align: 32, flags: DIFlagFwdDecl)
!18 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !16, producer: "clang", isOptimized: true, runtimeVersion: 2, emissionKind: 1, enums: !14, retainedTypes: !14, globals: !5, imports: !5)
!482 = !{i32 2, !"Debug Info Version", i32 3}
!634 = !DILocation(line: 143, column: 5, scope: !12)
