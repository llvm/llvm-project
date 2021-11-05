; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/function_metadata.prof -S | FileCheck %s
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/function_metadata.compact.afdo -S | FileCheck %s
; RUN: opt < %s -passes='pseudo-probe,thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/pseudo-probe-func-metadata.prof -S | FileCheck %s

;; Validate that with replay in effect, we import call sites even if they are below the threshold
;; Baseline import decisions
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-summary-hot-count=2000 -profile-file=%S/Inputs/function_metadata.prof -S | FileCheck -check-prefix=THRESHOLD %s
;; With replay decisions
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-summary-hot-count=2000 -profile-file=%S/Inputs/function_metadata.prof -S -sample-profile-inline-replay-scope=Function -sample-profile-inline-replay=%S/Inputs/function_metadata_replay.txt | FileCheck -check-prefix=THRESHOLD-REPLAY %s


; Tests whether the functions in the inline stack are added to the
; function_entry_count metadata.

declare void @foo()

declare void @bar()

declare !dbg !13 void @bar_dbg()

define void @bar_available() #0 !dbg !14 {
  ret void
}

; CHECK: define void @test({{.*}} !prof ![[ENTRY_TEST:[0-9]+]]
; THRESHOLD: define void @test({{.*}} !prof ![[ENTRY_THRESHOLD:[0-9]+]]
; THRESHOLD-REPLAY: define void @test({{.*}} !prof ![[ENTRY_REPLAY_TEST:[0-9]+]]
define void @test(void ()*) #0 !dbg !7 {
  %2 = alloca void ()*
  store void ()* %0, void ()** %2
  %3 = load void ()*, void ()** %2
  ; CHECK: call {{.*}}, !prof ![[PROF:[0-9]+]]
  call void @foo(), !dbg !18
  call void %3(), !dbg !19
  ret void
}

; CHECK: define void @test_liveness({{.*}} !prof ![[ENTRY_TEST_LIVENESS:[0-9]+]]
; THRESHOLD: define void @test_liveness({{.*}} !prof ![[ENTRY_THRESHOLD:[0-9]+]]
; THRESHOLD-REPLAY: define void @test_liveness({{.*}} !prof ![[ENTRY_REPLAY_TEST_LIVENESS:[0-9]+]]
define void @test_liveness() #0 !dbg !12 {
  call void @foo(), !dbg !20
  ret void
}

; GUIDs of foo, bar, foo1, foo2 and foo3 should be included in the metadata to
; make sure hot inline stacks are imported. The total count of baz is lower
; than the hot cutoff threshold and its GUID should not be included in the
; metadata.
; CHECK: ![[ENTRY_TEST]] = !{!"function_entry_count", i64 1, i64 2494702099028631698, i64 6699318081062747564, i64 7682762345278052905,  i64 -7908226060800700466, i64 -2012135647395072713}

; Check GUIDs for foo, bar and bar_dbg are included in the metadata to
; make sure the liveness analysis can capture the dependency from test_liveness
; to bar. bar_available should not be included as it's within the same module.
; CHECK: ![[ENTRY_TEST_LIVENESS]] = !{!"function_entry_count", i64 1, i64 6699318081062747564, i64 -2012135647395072713, i64 -1522495160813492905}

; With high threshold, nothing should be imported
; THRESHOLD: ![[ENTRY_THRESHOLD]] = !{!"function_entry_count", i64 1}

; With high threshold and replay, sites that are in the replay (foo, and transitively bar and bar_dbg in function test_liveness) should be imported
; THRESHOLD-REPLAY: ![[ENTRY_REPLAY_TEST]] = !{!"function_entry_count", i64 1}
; THRESHOLD-REPLAY: ![[ENTRY_REPLAY_TEST_LIVENESS]] = !{!"function_entry_count", i64 1, i64 6699318081062747564, i64 -2012135647395072713, i64 -1522495160813492905}

attributes #0 = {"use-sample-profile"}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: NoDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "calls.cc", directory: ".")
!2 = !{}
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "test", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 7, file: !1, scope: !1, type: !6, retainedNodes: !2)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5 "}
!12 = distinct !DISubprogram(name: "test_liveness", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 7, file: !1, scope: !1, type: !6, retainedNodes: !2)
!13 = !DISubprogram(name: "bar_dbg", scope: !1, file: !1, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!14 = distinct !DISubprogram(name: "bar_available", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 7, file: !1, scope: !1, type: !6, retainedNodes: !2)
!15 = !DILexicalBlockFile(discriminator: 1, file: !1, scope: !7)
!17 = distinct !DILexicalBlock(line: 10, column: 0, file: !1, scope: !7)
!18 = !DILocation(line: 10, scope: !17)
!19 = !DILocation(line: 11, scope: !17)
!20 = !DILocation(line: 8, scope: !12)
