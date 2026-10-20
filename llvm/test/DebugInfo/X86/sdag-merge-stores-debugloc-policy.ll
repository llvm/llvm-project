; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=finalize-isel -pick-merged-source-locations -o - %s | FileCheck %s --check-prefixes=CHECK,PICK

;; Check when to keep the lowest-addressed store's location and when to merge.
;; Merging may set the source line or an inlinedAt call-site line to 0.
;; -pick-merged-source-locations selects an original location when merging;
;; cases that keep the first store's location behave the same in both modes.

; Capture input locations from the IR printed before the machine functions.
; CHECK-LABEL: define void @second_store_inlined_in_first(
; CHECK: store i32 0, ptr %p, align 4, !dbg ![[OUTER_LOC:[0-9]+]]
; CHECK-LABEL: define void @first_store_inlined_second_in_outer(
; CHECK-SAME: !dbg ![[OUTER_SCOPE:[0-9]+]]
; CHECK: store i32 0, ptr %p, align 4, !dbg ![[INLINE_LOC:[0-9]+]]
; CHECK-LABEL: define void @same_inlined_func_two_callsites(
; CHECK-SAME: !dbg ![[TWO_CALLS_SCOPE:[0-9]+]]
; CHECK: store i32 0, ptr %q, align 4, !dbg ![[SECOND_CALL_LOC:[0-9]+]]
; CHECK-DAG: ![[INL_A:[0-9]+]] = distinct !DISubprogram(name: "inl_a",

;; The store at %p is on outer line 20; %q comes from inlined inl_a (line 10).
;; They merge into one 64-bit store. Keep %p's outer line 20 with or without
;; -pick-merged-source-locations, even though the inlined store has a smaller
;; line number.
; CHECK-LABEL: name: second_store_inlined_in_first
; CHECK: MOV64mi32 {{.*}}, debug-location ![[OUTER_LOC]] :: (store (s64)
define void @second_store_inlined_in_first(ptr %p) !dbg !10 {
  store i32 0, ptr %p, align 4, !dbg !11
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !13
  ret void
}
!10 = distinct !DISubprogram(name: "second_store_inlined_in_first", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocation(line: 20, scope: !10)
!12 = !DILocation(line: 30, scope: !10)
!13 = !DILocation(line: 10, scope: !5, inlinedAt: !12)

;; The store at %p comes from inlined inl_a (line 10); %q is on outer line 20.
;; The merged 64-bit store must use outer line 0 by default, without inl_a's
;; frame. With -pick-merged-source-locations, keep %p's location in inl_a.
; CHECK-LABEL: name: first_store_inlined_second_in_outer
; DEFAULT: MOV64mi32 {{.*}}, debug-location !DILocation(line: 0, scope: ![[OUTER_SCOPE]]) :: (store (s64)
; PICK: MOV64mi32 {{.*}}, debug-location ![[INLINE_LOC]] :: (store (s64)
define void @first_store_inlined_second_in_outer(ptr %p) !dbg !20 {
  store i32 0, ptr %p, align 4, !dbg !22
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !23
  ret void
}
!20 = distinct !DISubprogram(name: "first_store_inlined_second_in_outer", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 30, scope: !20)
!22 = !DILocation(line: 10, scope: !5, inlinedAt: !21)
!23 = !DILocation(line: 20, scope: !20)

;; Both stores are inl_a:10, inlined at outer lines 30 and 40 respectively.
;; The merged 64-bit store keeps inl_a:10 but uses outer call-site line 0.
;; With -pick-merged-source-locations, it keeps %q's call site at line 40.
; CHECK-LABEL: name: same_inlined_func_two_callsites
; DEFAULT: MOV64mi32 {{.*}}, debug-location !DILocation(line: 10, scope: ![[INL_A]], inlinedAt: !DILocation(line: 0, scope: ![[TWO_CALLS_SCOPE]])) :: (store (s64)
; PICK: MOV64mi32 {{.*}}, debug-location ![[SECOND_CALL_LOC]] :: (store (s64)
define void @same_inlined_func_two_callsites(ptr %p) !dbg !30 {
  store i32 0, ptr %p, align 4, !dbg !33
  %q = getelementptr i32, ptr %p, i64 1
  store i32 0, ptr %q, align 4, !dbg !34
  ret void
}
!30 = distinct !DISubprogram(name: "same_inlined_func_two_callsites", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocation(line: 30, scope: !30)
!32 = !DILocation(line: 40, scope: !30)
!33 = !DILocation(line: 10, scope: !5, inlinedAt: !31)
!34 = !DILocation(line: 10, scope: !5, inlinedAt: !32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "store-merge.c", directory: "/")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!5 = distinct !DISubprogram(name: "inl_a", scope: !1, file: !1, line: 1, type: !2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
