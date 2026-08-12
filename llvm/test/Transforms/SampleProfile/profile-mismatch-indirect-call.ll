; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/profile-mismatch-indirect-call.prof -report-profile-staleness -persist-profile-staleness -S 2>%t -o %t.ll
; RUN: FileCheck %s --input-file %t
; RUN: FileCheck %s --input-file %t.ll -check-prefix=CHECK-MD

; The profile records a direct call to "foo" at the location of the IR indirect
; call(line offset 1), which is not a mismatch as only one call target may be
; sampled. Only the "baz"(IR) vs "qux"(profile) callsite at line offset 3 is a
; real mismatch.

; CHECK: (1/3) of callsites' profile are invalid and (100/300) of samples are discarded due to callsite location mismatch.

; CHECK-MD: ![[#]] = !{!"NumMismatchedCallsites", i64 1, !"NumRecoveredCallsites", i64 0, !"TotalProfiledCallsites", i64 3, !"MismatchedCallsiteSamples", i64 100, !"RecoveredCallsiteSamples", i64 0}

define void @test(ptr %fp) #0 !dbg !3 {
entry:
  tail call void %fp(), !dbg !6
  tail call void @bar(), !dbg !7
  tail call void @baz(), !dbg !8
  ret void, !dbg !9
}

declare void @bar()

declare void @baz()

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", emissionKind: FullDebug, debugInfoForProfiling: true)
!1 = !DIFile(filename: "test.c", directory: "test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DILocation(line: 6, column: 3, scope: !3)
!7 = !DILocation(line: 7, column: 3, scope: !3)
!8 = !DILocation(line: 8, column: 3, scope: !3)
!9 = !DILocation(line: 9, column: 1, scope: !3)
