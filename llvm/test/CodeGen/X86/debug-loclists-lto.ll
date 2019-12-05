; RUN: llc -mtriple=x86_64-pc-linux -filetype=asm -function-sections < %s | FileCheck --implicit-check-not=loclists_table_base %s

; CHECK: {{^}}.Lloclists_table_base0:
; CHECK-NEXT: .long   .Ldebug_loc0-.Lloclists_table_base0
; CHECK-NEXT: .long   .Ldebug_loc1-.Lloclists_table_base0
; CHECK: .long   .Lloclists_table_base0  # DW_AT_loclists_base
; CHECK: .long   .Lloclists_table_base0  # DW_AT_loclists_base

; Function Attrs: uwtable
define dso_local void @_Z2f2v() local_unnamed_addr #0 !dbg !15 {
entry:
  tail call void @_Z2f1v(), !dbg !19
  call void @llvm.dbg.value(metadata i32 3, metadata !17, metadata !DIExpression()), !dbg !20
  tail call void @_Z2f1v(), !dbg !21
  ret void, !dbg !22
}
declare !dbg !4 dso_local void @_Z2f1v() local_unnamed_addr #1
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2
; Function Attrs: uwtable
define dso_local void @_Z2f3v() local_unnamed_addr #0 !dbg !23 {
entry:
  tail call void @_Z2f1v(), !dbg !26
  call void @llvm.dbg.value(metadata i32 3, metadata !25, metadata !DIExpression()), !dbg !27
  tail call void @_Z2f1v(), !dbg !28
  ret void, !dbg !29
}

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0, !7}
!llvm.ident = !{!11, !11}
!llvm.module.flags = !{!12, !13, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project.git 9b962d83ece841e43fd2823375dc6ddc94c1b178)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "loc1.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "3c96069dc8a3a1e7868038213ed0364a")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project.git 9b962d83ece841e43fd2823375dc6ddc94c1b178)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !9, nameTableKind: None)
!8 = !DIFile(filename: "loc2.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "2d309df0c6f5d8ce7264cc7696738fa9")
!9 = !{!10}
!10 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !8, file: !8, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!11 = !{!"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 9b962d83ece841e43fd2823375dc6ddc94c1b178)"}
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !5, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!16 = !{!17}
!17 = !DILocalVariable(name: "i", scope: !15, file: !1, line: 3, type: !18)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DILocation(line: 4, column: 3, scope: !15)
!20 = !DILocation(line: 0, scope: !15)
!21 = !DILocation(line: 6, column: 3, scope: !15)
!22 = !DILocation(line: 7, column: 1, scope: !15)
!23 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !8, file: !8, line: 2, type: !5, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7, retainedNodes: !24)
!24 = !{!25}
!25 = !DILocalVariable(name: "i", scope: !23, file: !8, line: 3, type: !18)
!26 = !DILocation(line: 4, column: 3, scope: !23)
!27 = !DILocation(line: 0, scope: !23)
!28 = !DILocation(line: 6, column: 3, scope: !23)
!29 = !DILocation(line: 7, column: 1, scope: !23)
