; REQUIRES: object-emission

; Verify that DWARF v5 debug_names omit names from CUs that opt-out.
; RUN: llc -mtriple x86_64-pc-linux -filetype=obj < %s \
; RUN:   | llvm-dwarfdump -debug-names - | FileCheck %s

; CHECK: CU count: 1

; Check that the one CU that is indexed has a non-zero.
; Avoid checking for a specific offset to make the test more resilient.
; CHECK: Compilation Unit offsets [
; CHECK-NEXT: CU[0]: 0x{{[0-9]*[1-9][0-9]*}}
; CHECK-NEXT: ]

define dso_local i32 @main() !dbg !9 {
entry:
  ret i32 0, !dbg !13
}

define dso_local void @_Z2f1v() !dbg !14 {
entry:
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 339438) (llvm/trunk 339448)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "foo.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "f881137628fc8dd673b761eb7a1e2432")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 8.0.0 (trunk 339438) (llvm/trunk 339448)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!4 = !DIFile(filename: "bar.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "ba8dae3bceaf6ef87728337164565a87")
!5 = !{!"clang version 8.0.0 (trunk 339438) (llvm/trunk 339448)"}
!6 = !{i32 2, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 1, column: 13, scope: !9)
!14 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !4, file: !4, line: 1, type: !15, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !3, retainedNodes: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !DILocation(line: 1, column: 12, scope: !14)
