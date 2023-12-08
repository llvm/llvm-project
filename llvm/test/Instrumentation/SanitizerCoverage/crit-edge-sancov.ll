; RUN: opt -passes='module(sancov-module)' -sanitizer-coverage-trace-pc \
; RUN: -sanitizer-coverage-level=3 %s -S -o - | FileCheck %s

; The edge between %entry and %for.inc.i is a critical edge.
; SanitizerCoveragePass must split this critical edge in order to track
; coverage of this edge. SanitizerCoveragePass will also insert calls to
; @__sanitizer_cov_trace_pc using the debug location from the predecessor's
; branch.  but, if the branch itself is missing debug info (say, by accident
; due to a bug in an earlier transform), we would fail a verifier check that
; verifies calls to functions with debug info themselves have debug info.
; The definition of @__sanitizer_cov_trace_pc may be visible during LTO.

; Of the below checks, we really only care that the calls to
; @__sanitizer_cov_trace_pc retain !dbg metadata.

define void @update_shadow(i1 %c) !dbg !3 {
; CHECK-LABEL: @update_shadow(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc() #[[ATTR0:[0-9]+]], !dbg [[DBG6:![0-9]+]]
; CHECK:       entry.for.inc.i_crit_edge:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc() #[[ATTR0]], !dbg [[DBG7:![0-9]+]]
; CHECK:       if.end22.i:
; CHECK-NEXT:    call void @__sanitizer_cov_trace_pc() #[[ATTR0]], !dbg [[DBG8:![0-9]+]]
; CHECK:       [[DBG6]] = !DILocation(line: 192, scope: !3)
; CHECK:       [[DBG7]] = !DILocation(line: 0, scope: !3)
; CHECK:       [[DBG8]] = !DILocation(line: 129, column: 2, scope: !3)
entry:
  br i1 %c, label %for.inc.i, label %if.end22.i

if.end22.i:                                       ; preds = %entry
  br label %for.inc.i, !dbg !8

for.inc.i:                                        ; preds = %if.end22.i, %entry
  ret void, !dbg !6
}

define void @__sanitizer_cov_trace_pc() !dbg !7{
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "kernel/cfi.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "update_shadow", scope: !1, file: !1, line: 190, type: !4, scopeLine: 192, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 223, column: 1, scope: !3)
!7 = distinct !DISubprogram(name: "__sanitizer_cov_trace_pc", scope: !1, file: !1, line: 200, type: !4, scopeLine: 200, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DILocation(line: 129, column: 2, scope: !3)
