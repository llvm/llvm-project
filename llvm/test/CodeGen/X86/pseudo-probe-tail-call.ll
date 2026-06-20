; Verify that pseudo-probe call probes on tail call is not considered dangling
; Source:
; int bar(int);
; int foo_tail(int x) {
;   # entry probe
;   # direct call probe
;   return bar(x); // tail call probe is dropped
; }

; RUN: llc -mtriple=x86_64-unknown-unknown -stop-after=pseudo-probe-inserter %s -o - | FileCheck %s

; CHECK-LABEL: name: foo_tail
; CHECK: PSEUDO_PROBE 3369713592223921200, 1, 0, 0
; CHECK-NEXT: PSEUDO_PROBE 3369713592223921200, 2, 2, 0
; CHECK-NEXT: TAILJMPd64 {{.*}} @bar

declare !dbg !26 i32 @bar(i32 noundef) local_unnamed_addr

define dso_local i32 @foo_tail(i32 noundef %x) local_unnamed_addr !dbg !15 {
entry:
  call void @llvm.pseudoprobe(i64 3369713592223921200, i64 1, i32 0, i64 -1), !dbg !22
  %call = tail call i32 @bar(i32 noundef %x), !dbg !23
  ret i32 %call, !dbg !25
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.pseudo_probe_desc = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "tail.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i64 3369713592223921200, i64 281479271677951, !"foo_tail"}
!15 = distinct !DISubprogram(name: "foo_tail", scope: !1, file: !1, line: 3, type: !16, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
!16 = !DISubroutineType(types: !17)
!17 = !{!18, !18}
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !DILocation(line: 4, column: 16, scope: !15)
!23 = !DILocation(line: 4, column: 12, scope: !24)
!24 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 455082007)
!25 = !DILocation(line: 4, column: 5, scope: !15)
!26 = !DISubprogram(name: "bar", scope: !1, file: !1, line: 1, type: !16, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)


