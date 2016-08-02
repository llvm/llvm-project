; RUN: opt %s -sample-profile -sample-profile-file=%S/Inputs/inline-hint.prof -pass-remarks=sample-profile -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: Applied cold hint to globally cold function '_Z7cold_fnRxi' with 0.1
define void @_Z7cold_fnRxi() !dbg !4 {
entry:
  ret void, !dbg !29
}

; CHECK: Applied inline hint to globally hot function '_Z6hot_fnRxi' with 70.0
define void @_Z6hot_fnRxi() #0 !dbg !10 {
entry:
  ret void, !dbg !38
}

!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!1 = !DIFile(filename: "inline-hint.cc", directory: ".")
!2 = !{}
!3 = !{!4, !10, !11, !14}
!4 = distinct !DISubprogram(name: "cold_fn", linkageName: "_Z7cold_fnRxi", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !9}
!7 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !8, size: 64, align: 64)
!8 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "hot_fn", linkageName: "_Z6hot_fnRxi", scope: !1, file: !1, line: 7, type: !5, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!11 = distinct !DISubprogram(name: "compute", linkageName: "_Z7computex", scope: !1, file: !1, line: 11, type: !12, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{!8, !8}
!14 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 21, type: !15, isLocal: false, isDefinition: true, scopeLine: 21, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!9}
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{!"clang version 3.8.0 (trunk 254067) (llvm/trunk 254079)"}
!29 = !DILocation(line: 5, column: 1, scope: !4)
!38 = !DILocation(line: 9, column: 1, scope: !10)
