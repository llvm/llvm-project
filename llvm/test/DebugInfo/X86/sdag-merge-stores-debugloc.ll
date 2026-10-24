; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck %s

;; Test that when loads or stores from different inlined calls are merged, the
;; widened access uses line 0 in the common caller rather than inheriting an
;; inlined scope that covers only part of it, while auxiliary value operations
;; keep their original debug locations.

; CHECK-DAG: ![[CALLER:[0-9]+]] = distinct !DISubprogram(name: "siblings",
; CHECK-DAG: ![[FIRST:[0-9]+]] = distinct !DISubprogram(name: "clear_first",
; CHECK-DAG: ![[SECOND:[0-9]+]] = distinct !DISubprogram(name: "clear_second",
; CHECK-DAG: ![[SAME_CALL:[0-9]+]] = !DILocation(line: 21, column: 3, scope: ![[#]])
; CHECK-DAG: ![[SAME:[0-9]+]] = !DILocation(line: 4, column: 3, scope: ![[FIRST]], inlinedAt: ![[SAME_CALL]])
; CHECK-DAG: ![[PARTIAL_CALL:[0-9]+]] = !DILocation(line: 41, column: 3, scope: ![[#]])
; CHECK-DAG: ![[PARTIAL:[0-9]+]] = !DILocation(line: 4, column: 3, scope: ![[FIRST]], inlinedAt: ![[PARTIAL_CALL]])
; CHECK-DAG: ![[EXTRACT:[0-9]+]] = distinct !DISubprogram(name: "extract_siblings",
; CHECK-DAG: ![[COPY:[0-9]+]] = distinct !DISubprogram(name: "copy_siblings",
; CHECK-DAG: ![[INDEPENDENT:[0-9]+]] = distinct !DISubprogram(name: "copy_independent_locations",
; CHECK-DAG: ![[LOAD_CALL:[0-9]+]] = !DILocation(line: 71, column: 3, scope: ![[INDEPENDENT]])
; CHECK-DAG: ![[LOAD:[0-9]+]] = !DILocation(line: 4, column: 3, scope: ![[FIRST]], inlinedAt: ![[LOAD_CALL]])
; CHECK-DAG: ![[STORE_CALL:[0-9]+]] = !DILocation(line: 72, column: 3, scope: ![[INDEPENDENT]])
; CHECK-DAG: ![[STORE:[0-9]+]] = !DILocation(line: 7, column: 3, scope: ![[SECOND]], inlinedAt: ![[STORE_CALL]])
; CHECK-DAG: ![[COPY_PARTIAL_CALL:[0-9]+]] = !DILocation(line: 81, column: 3, scope: ![[#]])
; CHECK-DAG: ![[COPY_PARTIAL:[0-9]+]] = !DILocation(line: 4, column: 3, scope: ![[FIRST]], inlinedAt: ![[COPY_PARTIAL_CALL]])
; CHECK-DAG: ![[ROTATE:[0-9]+]] = distinct !DISubprogram(name: "rotate_siblings",
; CHECK-DAG: ![[ROTATE_CALL:[0-9]+]] = !DILocation(line: 91, column: 3, scope: ![[ROTATE]])
; CHECK-DAG: ![[ROTATE_VALUE:[0-9]+]] = !DILocation(line: 4, column: 3, scope: ![[FIRST]], inlinedAt: ![[ROTATE_CALL]])

;; The two i32 stores come from the inlined functions clear_first and
;; clear_second. The merged 64-bit store uses line 0 in their caller, siblings.
; CHECK-LABEL: name: siblings
; CHECK: MOV64mi32 {{.*}}, debug-location !DILocation(line: 0, scope: ![[CALLER]]) :: (store (s64)
define void @siblings(ptr %p) !dbg !10 {
  store i32 0, ptr %p, align 4, !dbg !13
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !14
  ret void
}

;; Both i32 stores are on line 4 in clear_first, inlined at the same call site.
;; The merged 64-bit store keeps that line and call site.
; CHECK-LABEL: name: same_location
; CHECK: MOV64mi32 {{.*}}, debug-location ![[SAME]] :: (store (s64)
define void @same_location(ptr %p) !dbg !20 {
  store i32 0, ptr %p, align 4, !dbg !22
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !22
  ret void
}

;; Only the store at %p has a debug location; the store at %q has none.
;; The merged 64-bit store must have no debug location.
; CHECK-LABEL: name: missing_location
; CHECK: MOV64mi32
; CHECK-NOT: debug-location
; CHECK-SAME: :: (store (s64)
define void @missing_location(ptr %p) !dbg !30 {
  store i32 0, ptr %p, align 4, !dbg !32
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4
  ret void
}

;; The stores at %p and %q share a location in inlined clear_first. Merge them
;; into one 64-bit store with that location. The store at %r stays 32-bit; its
;; location in clear_second must not affect the merged store.
; CHECK-LABEL: name: partial_merge
; CHECK-DAG: MOV64mi32 {{.*}}, debug-location ![[PARTIAL]] :: (store (s64)
; CHECK-DAG: MOV32mi
define void @partial_merge(ptr %p) !dbg !40 {
  store i32 0, ptr %p, align 4, !dbg !43
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !43
  %r = getelementptr i32, ptr %p, i64 2
  store i32 0, ptr %r, align 4, !dbg !44
  ret void
}

;; The four stores of extracted i32 elements come from the inlined functions
;; clear_first and clear_second. The merged 128-bit store uses line 0 in their
;; common caller, extract_siblings.
; CHECK-LABEL: name: extract_siblings
; CHECK: debug-location !DILocation(line: 0, scope: ![[EXTRACT]]) :: (store (s128)
define void @extract_siblings(ptr %p, <4 x i32> %v) !dbg !50 {
  %a = extractelement <4 x i32> %v, i64 0
  %b = extractelement <4 x i32> %v, i64 1
  %c = extractelement <4 x i32> %v, i64 2
  %d = extractelement <4 x i32> %v, i64 3
  store i32 %a, ptr %p, align 16, !dbg !53
  %q = getelementptr i32, ptr %p, i64 1
  store i32 %b, ptr %q, align 4, !dbg !53
  %r = getelementptr i32, ptr %p, i64 2
  store i32 %c, ptr %r, align 8, !dbg !54
  %s = getelementptr i32, ptr %p, i64 3
  store i32 %d, ptr %s, align 4, !dbg !54
  ret void
}

;; The two i32 loads come from inlined clear_first and clear_second, as do
;; the two stores. The merged 64-bit load and store both use line 0 in their
;; common caller, copy_siblings.
; CHECK-LABEL: name: copy_siblings
; CHECK: MOV64rm {{.*}}, debug-location !DILocation(line: 0, scope: ![[COPY]]) :: (load (s64)
; CHECK: MOV64mr {{.*}}, debug-location !DILocation(line: 0, scope: ![[COPY]]) :: (store (s64)
define void @copy_siblings(ptr noalias %d, ptr noalias %s) !dbg !60 {
  %s1 = getelementptr i32, ptr %s, i64 1
  %d1 = getelementptr i32, ptr %d, i64 1
  %a = load i32, ptr %s, align 4, !dbg !63
  %b = load i32, ptr %s1, align 4, !dbg !64
  store i32 %a, ptr %d, align 4, !dbg !63
  store i32 %b, ptr %d1, align 4, !dbg !64
  ret void
}

;; Both i32 loads have the same location in inlined clear_first, and both
;; stores share a location in inlined clear_second. The merged 64-bit load
;; and store each keep their own location.
; CHECK-LABEL: name: copy_independent_locations
; CHECK: MOV64rm {{.*}}, debug-location ![[LOAD]] :: (load (s64)
; CHECK: MOV64mr {{.*}}, debug-location ![[STORE]] :: (store (s64)
define void @copy_independent_locations(ptr noalias %d, ptr noalias %s) !dbg !70 {
  %s1 = getelementptr i32, ptr %s, i64 1
  %d1 = getelementptr i32, ptr %d, i64 1
  %a = load i32, ptr %s, align 4, !dbg !73
  %b = load i32, ptr %s1, align 4, !dbg !73
  store i32 %a, ptr %d, align 4, !dbg !74
  store i32 %b, ptr %d1, align 4, !dbg !74
  ret void
}

;; The i32 copies from %s and %s1 share a location in inlined clear_first.
;; The merged 64-bit load and store keep that location. The copy from %s2
;; stays 32-bit and must not affect either merged location.
; CHECK-LABEL: name: copy_partial_merge
; CHECK-DAG: MOV64rm {{.*}}, debug-location ![[COPY_PARTIAL]] :: (load (s64)
; CHECK-DAG: MOV64mr {{.*}}, debug-location ![[COPY_PARTIAL]] :: (store (s64)
; CHECK-DAG: MOV32rm
; CHECK-DAG: MOV32mr
define void @copy_partial_merge(ptr noalias %d, ptr noalias %s) !dbg !80 {
  %s1 = getelementptr i32, ptr %s, i64 1
  %s2 = getelementptr i32, ptr %s, i64 2
  %d1 = getelementptr i32, ptr %d, i64 1
  %d2 = getelementptr i32, ptr %d, i64 2
  %a = load i32, ptr %s, align 4, !dbg !83
  %b = load i32, ptr %s1, align 4, !dbg !83
  %c = load i32, ptr %s2, align 4, !dbg !84
  store i32 %a, ptr %d, align 4, !dbg !83
  store i32 %b, ptr %d1, align 4, !dbg !83
  store i32 %c, ptr %d2, align 4, !dbg !84
  ret void
}

;; The loads come from inlined clear_first and clear_second; their values
;; are stored in reverse order. Expect a 64-bit load, a rotate by 32, and a
;; 64-bit store. Both memory operations use line 0 in rotate_siblings; the
;; rotate keeps the location of the original load at %s.
; CHECK-LABEL: name: rotate_siblings
; CHECK: MOV64rm {{.*}}, debug-location !DILocation(line: 0, scope: ![[ROTATE]]) :: (load (s64)
; CHECK: {{(ROL|ROR)64ri}} {{.*}}, 32, {{.*}}debug-location ![[ROTATE_VALUE]]{{$}}
; CHECK: MOV64mr {{.*}}, debug-location !DILocation(line: 0, scope: ![[ROTATE]]) :: (store (s64)
define void @rotate_siblings(ptr noalias %d, ptr noalias %s) !dbg !90 {
  %s1 = getelementptr i32, ptr %s, i64 1
  %d1 = getelementptr i32, ptr %d, i64 1
  %a = load i32, ptr %s, align 4, !dbg !93
  %b = load i32, ptr %s1, align 4, !dbg !94
  store i32 %b, ptr %d, align 4, !dbg !94
  store i32 %a, ptr %d1, align 4, !dbg !93
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "store-merge.c", directory: "/")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!5 = distinct !DISubprogram(name: "clear_first", scope: !1, file: !1, line: 3, type: !2, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = distinct !DISubprogram(name: "clear_second", scope: !1, file: !1, line: 6, type: !2, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = distinct !DISubprogram(name: "siblings", scope: !1, file: !1, line: 10, type: !2, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocation(line: 11, column: 3, scope: !10)
!12 = !DILocation(line: 12, column: 3, scope: !10)
!13 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !11)
!14 = !DILocation(line: 7, column: 3, scope: !6, inlinedAt: !12)
!20 = distinct !DISubprogram(name: "same_location", scope: !1, file: !1, line: 20, type: !2, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 21, column: 3, scope: !20)
!22 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !21)
!30 = distinct !DISubprogram(name: "missing_location", scope: !1, file: !1, line: 30, type: !2, scopeLine: 30, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocation(line: 31, column: 3, scope: !30)
!32 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !31)
!40 = distinct !DISubprogram(name: "partial_merge", scope: !1, file: !1, line: 40, type: !2, scopeLine: 40, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!41 = !DILocation(line: 41, column: 3, scope: !40)
!42 = !DILocation(line: 42, column: 3, scope: !40)
!43 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !41)
!44 = !DILocation(line: 7, column: 3, scope: !6, inlinedAt: !42)
!50 = distinct !DISubprogram(name: "extract_siblings", scope: !1, file: !1, line: 50, type: !2, scopeLine: 50, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = !DILocation(line: 51, column: 3, scope: !50)
!52 = !DILocation(line: 52, column: 3, scope: !50)
!53 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !51)
!54 = !DILocation(line: 7, column: 3, scope: !6, inlinedAt: !52)
!60 = distinct !DISubprogram(name: "copy_siblings", scope: !1, file: !1, line: 60, type: !2, scopeLine: 60, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!61 = !DILocation(line: 61, column: 3, scope: !60)
!62 = !DILocation(line: 62, column: 3, scope: !60)
!63 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !61)
!64 = !DILocation(line: 7, column: 3, scope: !6, inlinedAt: !62)
!70 = distinct !DISubprogram(name: "copy_independent_locations", scope: !1, file: !1, line: 70, type: !2, scopeLine: 70, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!71 = !DILocation(line: 71, column: 3, scope: !70)
!72 = !DILocation(line: 72, column: 3, scope: !70)
!73 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !71)
!74 = !DILocation(line: 7, column: 3, scope: !6, inlinedAt: !72)
!80 = distinct !DISubprogram(name: "copy_partial_merge", scope: !1, file: !1, line: 80, type: !2, scopeLine: 80, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!81 = !DILocation(line: 81, column: 3, scope: !80)
!82 = !DILocation(line: 82, column: 3, scope: !80)
!83 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !81)
!84 = !DILocation(line: 7, column: 3, scope: !6, inlinedAt: !82)
!90 = distinct !DISubprogram(name: "rotate_siblings", scope: !1, file: !1, line: 90, type: !2, scopeLine: 90, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!91 = !DILocation(line: 91, column: 3, scope: !90)
!92 = !DILocation(line: 92, column: 3, scope: !90)
!93 = !DILocation(line: 4, column: 3, scope: !5, inlinedAt: !91)
!94 = !DILocation(line: 7, column: 3, scope: !6, inlinedAt: !92)
