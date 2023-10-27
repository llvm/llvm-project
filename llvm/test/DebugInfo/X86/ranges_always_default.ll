; RUN: llc -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - -minimize-addr-in-v5=Default \
; RUN:   -split-dwarf-file=test.dwo \
; RUN:   | llvm-dwarfdump -debug-info -debug-addr -debug-rnglists -v - \
; RUN:   | FileCheck --check-prefix=RANGE %s

; RUN: llc -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - -minimize-addr-in-v5=Disabled \
; RUN:   -split-dwarf-file=test.dwo \
; RUN:   | llvm-dwarfdump -debug-info -debug-addr -debug-rnglists -v - \
; RUN:   | FileCheck --check-prefix=NORANGE %s

; A simpler example than used in ranges_always.ll, since this doesn't test all
; the nuances of where minimizing ranges are useful. This is only testing the
; defaulting behavior - specifically that the "ranges" version of the
; functionality is used when Split DWARF emission occurs (so, when split dwarf is
; enabled and !(gmlt+!split-dwarf-inlining) (because if the latter is true, then
; split dwarf emission doesn't occur because it'd be redundant/extra verbose)

; CHECK: DW_TAG_inlined_subroutine
; CHECK-NOT: {{DW_TAG|NULL}}
; RANGE: DW_AT_ranges
; CHECK-NOT: {{DW_TAG|NULL}}
; NORANGE: DW_AT_low_pc
; CHECK-NOT: {{DW_TAG|NULL}}
; NORANGE: DW_AT_high_pc
; CHECK: NULL


; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z2f3v() !dbg !10 {
entry:
  call void @_Z2f1v(), !dbg !13
  call void @_Z2f1v(), !dbg !14
  ret void, !dbg !17
}

declare void @_Z2f1v()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0 (git@github.com:llvm/llvm-project.git 22afe19ac03f5b5db642cbb8ba7022c2ffc09710)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/proc/self/cwd", checksumkind: CSK_MD5, checksum: "77606e6e313660c9c1dac8290849946d")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 17.0.0 (git@github.com:llvm/llvm-project.git 22afe19ac03f5b5db642cbb8ba7022c2ffc09710)"}
!10 = distinct !DISubprogram(name: "f3", scope: !1, file: !1, line: 5, type: !11, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 6, column: 3, scope: !10)
!14 = !DILocation(line: 3, column: 3, scope: !15, inlinedAt: !16)
!15 = distinct !DISubprogram(name: "f2", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!16 = distinct !DILocation(line: 7, column: 3, scope: !10)
!17 = !DILocation(line: 8, column: 1, scope: !10)
