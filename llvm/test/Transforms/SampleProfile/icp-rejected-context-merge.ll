; RUN: opt -passes=sample-profile -sample-profile-use-preinliner -sample-profile-file=%S/Inputs/icp-rejected-context-merge.prof -S %s | FileCheck %s
; RUN: opt -passes=sample-profile -sample-profile-use-preinliner -sample-profile-file=%S/Inputs/icp-rejected-context-merge.prof -mtriple=x86_64-unknown-linux-gnu -S %s | FileCheck %s
;
; Only first promotes at each indirect call. Both hot rejected targets recover
; their standalone profiles, while cold does not. The two calls share a probe
; location, so each rejected context must be merged once, not twice.
;
; CHECK-LABEL: define i32 @first(
; CHECK-SAME: !prof ![[FIRST:[0-9]+]]
; CHECK-LABEL: define i32 @hot20(
; CHECK-SAME: !prof ![[HOT20:[0-9]+]]
; CHECK-LABEL: define i32 @hot10(
; CHECK-SAME: !prof ![[HOT10:[0-9]+]]
; CHECK-LABEL: define i32 @cold(
; CHECK-SAME: !prof ![[FIRST]]
; CHECK-LABEL: define i32 @dispatch(
; CHECK-COUNT-2: icmp eq ptr {{.*}}, @first
; CHECK-NOT: icmp eq ptr
; CHECK: ret i32
; CHECK-DAG: ![[FIRST]] = !{!"function_entry_count", i64 1}
; CHECK-DAG: ![[HOT20]] = !{!"function_entry_count", i64 20001}
; CHECK-DAG: ![[HOT10]] = !{!"function_entry_count", i64 10001}

target triple = "aarch64-unknown-linux-gnu"

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

define i32 @first(i32 %value) #0 !dbg !10 {
  call void @llvm.pseudoprobe(i64 -8202922476773768053, i64 1, i32 0, i64 -1), !dbg !20
  %result = add i32 %value, 1
  ret i32 %result
}

define i32 @hot20(i32 %value) #0 !dbg !11 {
  call void @llvm.pseudoprobe(i64 -4491665821553114269, i64 1, i32 0, i64 -1), !dbg !21
  %result = add i32 %value, 2
  ret i32 %result
}

define i32 @hot10(i32 %value) #0 !dbg !12 {
  call void @llvm.pseudoprobe(i64 -5298124545790704069, i64 1, i32 0, i64 -1), !dbg !22
  %result = add i32 %value, 3
  ret i32 %result
}

define i32 @cold(i32 %value) #0 !dbg !13 {
  call void @llvm.pseudoprobe(i64 -6778568560291945099, i64 1, i32 0, i64 -1), !dbg !23
  %result = add i32 %value, 4
  ret i32 %result
}

define i32 @dispatch(ptr %target, i32 %value) #0 !dbg !14 {
  call void @llvm.pseudoprobe(i64 -7762558725055652451, i64 1, i32 0, i64 -1), !dbg !24
  %a = call i32 %target(i32 %value), !dbg !26
  %b = call i32 %target(i32 %a), !dbg !26
  ret i32 %b
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!llvm.pseudo_probe_desc = !{!30, !31, !32, !33, !34}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!10 = distinct !DISubprogram(name: "first", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!20 = !DILocation(line: 1, column: 1, scope: !10)
!30 = !{i64 -8202922476773768053, i64 4294967295, !"first"}
!11 = distinct !DISubprogram(name: "hot20", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 1, column: 1, scope: !11)
!31 = !{i64 -4491665821553114269, i64 4294967295, !"hot20"}
!12 = distinct !DISubprogram(name: "hot10", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!22 = !DILocation(line: 1, column: 1, scope: !12)
!32 = !{i64 -5298124545790704069, i64 4294967295, !"hot10"}
!13 = distinct !DISubprogram(name: "cold", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!23 = !DILocation(line: 1, column: 1, scope: !13)
!33 = !{i64 -6778568560291945099, i64 4294967295, !"cold"}
!14 = distinct !DISubprogram(name: "dispatch", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = !DILocation(line: 1, column: 1, scope: !14)
!34 = !{i64 -7762558725055652451, i64 281479271677951, !"dispatch"}
!25 = !DILexicalBlockFile(scope: !14, file: !1, discriminator: 387973143)
!26 = !DILocation(line: 1, column: 1, scope: !25)
