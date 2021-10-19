; RUN: llc -mtriple=x86_64-linux -split-dwarf-file=foo.dwo -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; CHECK: DW_AT_name ("b.cpp")
; CHECK: DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_linkage_name      ("_ZN5outer2f2Ev")
; CHECK-NEXT:   DW_AT_name      ("f2")
; CHECK-NEXT:   DW_AT_decl_file (0x02)
; CHECK-NEXT:   DW_AT_decl_line (4)

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define dso_local void @_Z2f1v() local_unnamed_addr #0 !dbg !12 {
entry:
  ret void, !dbg !15
}

; Function Attrs: nounwind uwtable mustprogress
define dso_local void @_ZN5outer2f2Ev() local_unnamed_addr #1 align 2 !dbg !16 {
entry:
  tail call void @_Z2f1v(), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nounwind uwtable mustprogress
define dso_local void @_Z2f2v() local_unnamed_addr #1 !dbg !23 {
entry:
  tail call void @_Z2f1v(), !dbg !24
  ret void, !dbg !25
}

; Function Attrs: norecurse nounwind uwtable mustprogress
define dso_local i32 @main() local_unnamed_addr #2 !dbg !26 {
entry:
  tail call void @_Z2f1v() #3, !dbg !28
  tail call void @_Z2f1v() #3, !dbg !30
  ret i32 0, !dbg !32
}

attributes #0 = { noinline nounwind optnone uwtable mustprogress "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind uwtable mustprogress "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { norecurse nounwind uwtable mustprogress "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (git@github.com:llvm/llvm-project.git 9aa951e80e72decd95c7d972e1e0dde24260d336)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "a.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, producer: "clang version 13.0.0 (git@github.com:llvm/llvm-project.git 9aa951e80e72decd95c7d972e1e0dde24260d336)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!4 = !DIFile(filename: "b.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!5 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project.git 9aa951e80e72decd95c7d972e1e0dde24260d336)"}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{i32 1, !"ThinLTO", i32 0}
!11 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!12 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 2, type: !13, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DILocation(line: 3, column: 1, scope: !12)
!16 = distinct !DISubprogram(name: "f2", linkageName: "_ZN5outer2f2Ev", scope: !17, file: !1, line: 4, type: !13, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !20, retainedNodes: !2)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "outer", file: !18, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !19, identifier: "_ZTS5outer")
!18 = !DIFile(filename: "./a.h", directory: "/usr/local/google/home/blaikie/dev/scratch")
!19 = !{!20}
!20 = !DISubprogram(name: "f2", linkageName: "_ZN5outer2f2Ev", scope: !17, file: !18, line: 2, type: !13, scopeLine: 2, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: DISPFlagOptimized)
!21 = !DILocation(line: 5, column: 3, scope: !16)
!22 = !DILocation(line: 6, column: 1, scope: !16)
!23 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 7, type: !13, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 8, column: 3, scope: !23)
!25 = !DILocation(line: 9, column: 1, scope: !23)
!26 = distinct !DISubprogram(name: "main", scope: !4, file: !4, line: 2, type: !27, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !3, retainedNodes: !2)
!27 = !DISubroutineType(types: !2)
!28 = !DILocation(line: 5, column: 3, scope: !16, inlinedAt: !29)
!29 = distinct !DILocation(line: 3, column: 3, scope: !26)
!30 = !DILocation(line: 8, column: 3, scope: !23, inlinedAt: !31)
!31 = distinct !DILocation(line: 4, column: 3, scope: !26)
!32 = !DILocation(line: 5, column: 1, scope: !26)
