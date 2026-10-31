; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=finalize-isel -pick-merged-source-locations -o - %s | FileCheck %s --check-prefixes=CHECK,PICK

;; Test that when consecutive loads or stores are merged, we keep the first
;; operation's debug location if all merged operations are in the same function
;; or inlined call (or in functions inlined into it). Otherwise, merge their
;; locations with getMergedLocation, where -pick-merged-source-locations takes
;; effect; if any merged operation has no debug location, the result has none.

; CHECK-DAG: ![[HELPER:[0-9]+]] = distinct !DISubprogram(name: "helper",
; CHECK-DAG: ![[PLAIN:[0-9]+]] = distinct !DISubprogram(name: "plain_lines",
; CHECK-DAG: ![[PLAIN_FIRST:[0-9]+]] = !DILocation(line: 20, column: 3, scope: ![[PLAIN]])
; CHECK-DAG: ![[EXTRACT_CALL:[0-9]+]] = !DILocation(line: 110, column: 3, scope: ![[#]])
; CHECK-DAG: ![[EXTRACT_FIRST:[0-9]+]] = !DILocation(line: 20, column: 3, scope: ![[HELPER]], inlinedAt: ![[EXTRACT_CALL]])
; CHECK-DAG: ![[CALLER_FIRST:[0-9]+]] = !DILocation(line: 210, column: 3, scope: ![[#]])
; CHECK-DAG: ![[HELPER_CALLER:[0-9]+]] = distinct !DISubprogram(name: "helper_first",
; CHECK-DAG: ![[HELPER_CALL:[0-9]+]] = !DILocation(line: 320, column: 3, scope: ![[HELPER_CALLER]])
; CHECK-DAG: ![[HELPER_PICK:[0-9]+]] = !DILocation(line: 10, column: 3, scope: ![[HELPER]], inlinedAt: ![[HELPER_CALL]])
; CHECK-DAG: ![[REPEAT:[0-9]+]] = distinct !DISubprogram(name: "same_helper_two_calls",
; CHECK-DAG: ![[REPEAT_CALL:[0-9]+]] = !DILocation(line: 420, column: 3, scope: ![[REPEAT]])
; CHECK-DAG: ![[REPEAT_PICK:[0-9]+]] = !DILocation(line: 10, column: 3, scope: ![[HELPER]], inlinedAt: ![[REPEAT_CALL]])
; CHECK-DAG: ![[COPY:[0-9]+]] = distinct !DISubprogram(name: "copy_locations",
; CHECK-DAG: ![[COPY_CALL:[0-9]+]] = !DILocation(line: 510, column: 3, scope: ![[COPY]])
; CHECK-DAG: ![[COPY_FIRST:[0-9]+]] = !DILocation(line: 20, column: 3, scope: ![[HELPER]], inlinedAt: ![[COPY_CALL]])
; CHECK-DAG: ![[MERGED_CALLER:[0-9]+]] = distinct !DISubprogram(name: "merged_callsite",
; CHECK-DAG: ![[MERGED_CALL:[0-9]+]] = !DILocation(line: 0, scope: ![[MERGED_CALLER]])
; CHECK-DAG: ![[OTHER_CALL:[0-9]+]] = !DILocation(line: 710, column: 3, scope: ![[MERGED_CALLER]])
; CHECK-DAG: ![[MERGED_PICK:[0-9]+]] = !DILocation(line: 10, column: 3, scope: ![[HELPER]], inlinedAt: ![[OTHER_CALL]])
; CHECK-DAG: ![[NESTED_CALL:[0-9]+]] = !DILocation(line: 810, column: 3, scope: ![[#]])
; CHECK-DAG: ![[NESTED_FIRST:[0-9]+]] = !DILocation(line: 20, column: 3, scope: ![[HELPER]], inlinedAt: ![[NESTED_CALL]])

;; The two i32 stores are on lines 20 and 10 of plain_lines. Keep %p's line 20
;; on the merged 64-bit store, even with -pick-merged-source-locations.
; CHECK-LABEL: name: plain_lines
; CHECK: MOV64mi32 {{.*}}, debug-location ![[PLAIN_FIRST]] :: (store (s64)
define void @plain_lines(ptr %p) !dbg !100 {
  store i32 0, ptr %p, align 4, !dbg !101
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !102
  ret void
}

;; Four stores of extracted i32 elements come from the inlined function helper
;; at lines 20 and 10, inlined at the same call site. The merged 128-bit store
;; keeps %p's line 20, even with -pick-merged-source-locations.
; CHECK-LABEL: name: extract_inline_lines
; CHECK: debug-location ![[EXTRACT_FIRST]] :: (store (s128)
define void @extract_inline_lines(ptr %p, <4 x i32> %v) !dbg !110 {
  %a = extractelement <4 x i32> %v, i64 0
  %b = extractelement <4 x i32> %v, i64 1
  %c = extractelement <4 x i32> %v, i64 2
  %d = extractelement <4 x i32> %v, i64 3
  store i32 %a, ptr %p, align 16, !dbg !112
  %q = getelementptr i32, ptr %p, i64 1
  store i32 %b, ptr %q, align 4, !dbg !112
  %r = getelementptr i32, ptr %p, i64 2
  store i32 %c, ptr %r, align 8, !dbg !113
  %s = getelementptr i32, ptr %p, i64 3
  store i32 %d, ptr %s, align 4, !dbg !113
  ret void
}

;; The first store (at %p) is in the outer function caller_first at line 210,
;; while the second store (at %q) comes from the inlined function helper.
;; Select %p's line 210 in the outer function for the merged 64-bit store,
;; even with -pick-merged-source-locations.
; CHECK-LABEL: name: caller_first
; CHECK: MOV64mi32 {{.*}}, debug-location ![[CALLER_FIRST]] :: (store (s64)
define void @caller_first(ptr %p) !dbg !120 {
  store i32 0, ptr %p, align 4, !dbg !121
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !123
  ret void
}

;; The first store (at %p) comes from the inlined function helper, while the
;; second store (at %q) is in the outer function helper_first. Because %p's
;; inlined scope does not cover %q, the merged 64-bit store uses line 0 in
;; helper_first by default, or %p's location with -pick-merged-source-locations.
; CHECK-LABEL: name: helper_first
; DEFAULT: MOV64mi32 {{.*}}, debug-location !DILocation(line: 0, scope: ![[HELPER_CALLER]]) :: (store (s64)
; PICK: MOV64mi32 {{.*}}, debug-location ![[HELPER_PICK]] :: (store (s64)
define void @helper_first(ptr %p) !dbg !130 {
  store i32 0, ptr %p, align 4, !dbg !133
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !131
  ret void
}

;; Both stores are at helper line 10, inlined at caller lines 410 and 420.
;; The merged 64-bit store keeps helper line 10 with call-site line 0.
;; With -pick-merged-source-locations, keep %q's call site at line 420.
; CHECK-LABEL: name: same_helper_two_calls
; DEFAULT: MOV64mi32 {{.*}}, debug-location !DILocation(line: 10, column: 3, scope: ![[HELPER]], inlinedAt: !DILocation(line: 0, scope: ![[REPEAT]])) :: (store (s64)
; PICK: MOV64mi32 {{.*}}, debug-location ![[REPEAT_PICK]] :: (store (s64)
define void @same_helper_two_calls(ptr %p) !dbg !140 {
  store i32 0, ptr %p, align 4, !dbg !143
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !144
  ret void
}

;; The loads are at helper lines 20 and 10, inlined at the same call site.
;; The merged 64-bit load keeps %s's line 20. The stores are in the inlined
;; functions helper and other_helper, so the merged store uses line 0 in
;; copy_locations, or %d's location with -pick-merged-source-locations.
; CHECK-LABEL: name: copy_locations
; CHECK: MOV64rm {{.*}}, debug-location ![[COPY_FIRST]] :: (load (s64)
; DEFAULT: MOV64mr {{.*}}, debug-location !DILocation(line: 0, scope: ![[COPY]]) :: (store (s64)
; PICK: MOV64mr {{.*}}, debug-location ![[COPY_FIRST]] :: (store (s64)
define void @copy_locations(ptr noalias %d, ptr noalias %s) !dbg !150 {
  %s1 = getelementptr i32, ptr %s, i64 1
  %d1 = getelementptr i32, ptr %d, i64 1
  %a = load i32, ptr %s, align 4, !dbg !152
  %b = load i32, ptr %s1, align 4, !dbg !153
  store i32 %a, ptr %d, align 4, !dbg !152
  store i32 %b, ptr %d1, align 4, !dbg !155
  ret void
}

;; The store at %p has no debug location; the store at %q does. The merged
;; 64-bit store has none, even with -pick-merged-source-locations.
; CHECK-LABEL: name: missing_first
; CHECK: MOV64mi32
; CHECK-NOT: debug-location
; CHECK-SAME: :: (store (s64)
define void @missing_first(ptr %p) !dbg !160 {
  store i32 0, ptr %p, align 4
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !161
  ret void
}

;; Both stores are in the inlined function helper (lines 20 and 10), but their
;; inlinedAt locations are different call sites (!171 at line 0 and !172 at
;; line 710). Merging those call sites yields !171, which is identical to the
;; first store's inlinedAt node, but the merged 64-bit store must still merge
;; the two helper lines to line 0 (or pick %q's location with
;; -pick-merged-source-locations).
; CHECK-LABEL: name: merged_callsite
; DEFAULT: MOV64mi32 {{.*}}, debug-location !DILocation(line: 0, scope: ![[HELPER]], inlinedAt: ![[MERGED_CALL]]) :: (store (s64)
; PICK: MOV64mi32 {{.*}}, debug-location ![[MERGED_PICK]] :: (store (s64)
define void @merged_callsite(ptr %p) !dbg !170 {
  store i32 0, ptr %p, align 4, !dbg !173
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !174
  ret void
}

;; These two stores are merged, and the second store (at %q, in other_helper)
;; has a deeper inlining chain than the first (at %p, in helper) under the
;; same call site (!181). Keep %p's line 20 on the merged 64-bit store, even
;; with -pick-merged-source-locations.
; CHECK-LABEL: name: nested_callee
; CHECK: MOV64mi32 {{.*}}, debug-location ![[NESTED_FIRST]] :: (store (s64)
define void @nested_callee(ptr %p) !dbg !180 {
  store i32 0, ptr %p, align 4, !dbg !182
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !184
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "store-merge.c", directory: "/")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!10 = distinct !DISubprogram(name: "helper", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = distinct !DISubprogram(name: "other_helper", scope: !1, file: !1, line: 5, type: !2, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!100 = distinct !DISubprogram(name: "plain_lines", scope: !1, file: !1, line: 9, type: !2, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!101 = !DILocation(line: 20, column: 3, scope: !100)
!102 = !DILocation(line: 10, column: 3, scope: !100)
!110 = distinct !DISubprogram(name: "extract_inline_lines", scope: !1, file: !1, line: 100, type: !2, scopeLine: 100, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!111 = !DILocation(line: 110, column: 3, scope: !110)
!112 = !DILocation(line: 20, column: 3, scope: !10, inlinedAt: !111)
!113 = !DILocation(line: 10, column: 3, scope: !10, inlinedAt: !111)
!120 = distinct !DISubprogram(name: "caller_first", scope: !1, file: !1, line: 200, type: !2, scopeLine: 200, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!121 = !DILocation(line: 210, column: 3, scope: !120)
!122 = !DILocation(line: 220, column: 3, scope: !120)
!123 = !DILocation(line: 10, column: 3, scope: !10, inlinedAt: !122)
!130 = distinct !DISubprogram(name: "helper_first", scope: !1, file: !1, line: 300, type: !2, scopeLine: 300, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!131 = !DILocation(line: 310, column: 3, scope: !130)
!132 = !DILocation(line: 320, column: 3, scope: !130)
!133 = !DILocation(line: 10, column: 3, scope: !10, inlinedAt: !132)
!140 = distinct !DISubprogram(name: "same_helper_two_calls", scope: !1, file: !1, line: 400, type: !2, scopeLine: 400, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!141 = !DILocation(line: 410, column: 3, scope: !140)
!142 = !DILocation(line: 420, column: 3, scope: !140)
!143 = !DILocation(line: 10, column: 3, scope: !10, inlinedAt: !141)
!144 = !DILocation(line: 10, column: 3, scope: !10, inlinedAt: !142)
!150 = distinct !DISubprogram(name: "copy_locations", scope: !1, file: !1, line: 500, type: !2, scopeLine: 500, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!151 = !DILocation(line: 510, column: 3, scope: !150)
!152 = !DILocation(line: 20, column: 3, scope: !10, inlinedAt: !151)
!153 = !DILocation(line: 10, column: 3, scope: !10, inlinedAt: !151)
!154 = !DILocation(line: 520, column: 3, scope: !150)
!155 = !DILocation(line: 30, column: 3, scope: !11, inlinedAt: !154)
!160 = distinct !DISubprogram(name: "missing_first", scope: !1, file: !1, line: 600, type: !2, scopeLine: 600, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!161 = !DILocation(line: 610, column: 3, scope: !160)
!170 = distinct !DISubprogram(name: "merged_callsite", scope: !1, file: !1, line: 700, type: !2, scopeLine: 700, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!171 = !DILocation(line: 0, scope: !170)
!172 = !DILocation(line: 710, column: 3, scope: !170)
!173 = !DILocation(line: 20, column: 3, scope: !10, inlinedAt: !171)
!174 = !DILocation(line: 10, column: 3, scope: !10, inlinedAt: !172)
!180 = distinct !DISubprogram(name: "nested_callee", scope: !1, file: !1, line: 800, type: !2, scopeLine: 800, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!181 = !DILocation(line: 810, column: 3, scope: !180)
!182 = !DILocation(line: 20, column: 3, scope: !10, inlinedAt: !181)
!183 = !DILocation(line: 30, column: 3, scope: !10, inlinedAt: !181)
!184 = !DILocation(line: 10, column: 3, scope: !11, inlinedAt: !183)
